# import the necessary packages

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import cv2, os



from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from keras.preprocessing import image



from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense

from keras.layers import Dropout

from keras.layers import Activation

from keras.models import model_from_json



from keras.layers import BatchNormalization

from keras.layers import Input

from keras.layers import GlobalAveragePooling2D

from keras.applications.resnet50 import ResNet50

from keras.applications.resnet50 import preprocess_input

from keras.models import Model

from keras.optimizers import rmsprop

import keras

import gc
df_train = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')

df_test = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')



x = df_train['id_code']

y = df_train['diagnosis']



unique, counts = np.unique(y, return_counts=True)



plt.subplot()

plt.hist(y)

plt.show()



plt.subplot()

plt.pie(counts, labels=unique, autopct='%1.1f%%', startangle=90)

plt.show()



# y = to_categorical(y, num_classes=5)

y = LabelEncoder().fit_transform(y)



SIZE = 256



train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.2, stratify=y)

print(train_x.shape, train_y.shape, valid_x.shape, valid_y.shape)
def crop_image_from_gray(img,tol=7):

    if img.ndim ==2:

        mask = img>tol

        return img[np.ix_(mask.any(1),mask.any(0))]

    elif img.ndim==3:

        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        mask = gray_img>tol

        

        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]

        if (check_shape == 0): # image is too dark so that we crop out everything,

            return img # return original image

        else:

            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]

            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]

            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]

            img = np.stack([img1,img2,img3],axis=-1)

        return img

    

# preprocessing function

def preprocess(img):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = crop_image_from_gray(img)

    img = cv2.resize(img, (SIZE, SIZE))

    img = cv2.addWeighted(img,4, cv2.GaussianBlur(img ,(0,0), 30) ,-4 ,128)

    return img
fig = plt.figure(figsize=(25, 16))

for class_id in sorted(np.unique(train_y)):

    for i, (idx, row) in enumerate(df_train.loc[df_train['diagnosis'] == class_id].sample(7).iterrows()):

        ax = fig.add_subplot(5, 7, class_id * 7 + i + 1, xticks=[], yticks=[])

        path=f"../input/aptos2019-blindness-detection/train_images/{row['id_code']}.png"

        img = preprocess(cv2.imread(path))



        plt.imshow(img)

        ax.set_title('%d-%d-%s' % (class_id, idx, row['id_code']) )
train_x_ = []

for i in range(len(train_x)):

    path = os.path.join('../input/aptos2019-blindness-detection/train_images/', train_x[train_x.index[i]]+'.png')

    train_x_.append(path)



valid_x_ = []

for i in range(len(valid_x)):

    path = os.path.join('../input/aptos2019-blindness-detection/train_images/', valid_x[valid_x.index[i]]+'.png')

    valid_x_.append(path)



test = []

for i in range(len(df_test['id_code'])):

    path = os.path.join('../input/aptos2019-blindness-detection/test_images/', df_test['id_code'][df_test['id_code'].index[i]]+'.png')

    test.append(path)
# construct image generator for data augmentation

train_datagen = image.ImageDataGenerator(rotation_range=30,

                                         zoom_range=0.25,

                                         width_shift_range=0.2, 

                                         height_shift_range=0.2, 

                                         shear_range=0.15,

                                         horizontal_flip=True, 

                                         vertical_flip=True,

                                         fill_mode="nearest",

                                         preprocessing_function=preprocess,

                                         rescale=1./255)



# train the network



train_path = pd.concat([pd.Series(train_x_), pd.Series(train_y)], axis=1).rename({0:'path',1:'diagnosis'}, axis=1)

valid_path = pd.concat([pd.Series(valid_x_), pd.Series(valid_y)], axis=1).rename({0:'path',1:'diagnosis'}, axis=1)



train_path['diagnosis'] = train_path['diagnosis'].astype(str)

valid_path['diagnosis'] = valid_path['diagnosis'].astype(str)



#fit model

gen_train = train_datagen.flow_from_dataframe(train_path, x_col='path', y_col='diagnosis', target_size=(SIZE,SIZE), class_mode='categorical')

gen_valid = train_datagen.flow_from_dataframe(valid_path, x_col='path', y_col='diagnosis', target_size=(SIZE,SIZE), class_mode='categorical')



steps_train = gen_train.n//gen_train.batch_size

steps_valid = gen_valid.n//gen_valid.batch_size
def create_resnet(img_dim):

    input_tensor=Input(shape=(img_dim, img_dim,3))

    base_model = ResNet50(weights=None,

                          include_top=False,

                          input_tensor=input_tensor)

    base_model.load_weights('../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')

    x=GlobalAveragePooling2D()(base_model.output)

    x=Dropout(0.3)(x)

    x=Dense(2048, activation='relu')(x)

    x=Dropout(0.3)(x)

    x=Dense(512, activation='relu')(x)

    x=Dropout(0.3)(x)

    x=Dense(128, activation='relu')(x)

    x=Dropout(0.3)(x)

    x=BatchNormalization()(x)

    output_layer=Dense(5,activation='softmax', name="Output_Layer")(x)

    model_resnet=Model(input_tensor, output_layer)

    return model_resnet



model_resnet=create_resnet(SIZE)



for layers in model_resnet.layers:

    layers.trainable=True

    

model_resnet.summary()
lr = 1e-3

optimizer=rmsprop(lr=lr,decay=0.2)

model_resnet.compile(optimizer=optimizer, loss=keras.losses.categorical_crossentropy,  metrics=['accuracy'])
history=model_resnet.fit_generator(generator=gen_train,

                        steps_per_epoch=steps_train,

                        validation_data=gen_valid,

                        validation_steps=steps_valid,

                        epochs=25,

                        class_weight=counts)



gc.collect()
plt.figure(figsize=(8, 8))

plt.title("Learning curve")

plt.plot(history.history["loss"], label="loss")

plt.plot(history.history["val_loss"], label="val_loss")

plt.plot( np.argmin(history.history["val_loss"]), np.min(history.history["val_loss"]), marker="x", color="r", label="best model")

plt.xlabel("Epochs")

plt.ylabel("log_loss")

plt.legend()
# making predictions



prediction = []

sample = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')



for path in test:

    img = preprocess(cv2.imread(path))

    score_predict = model_resnet.predict((img[np.newaxis])/255.)

    label_predict = np.argmax(score_predict)

    prediction.append(str(label_predict))

    

sample['diagnosis'] = prediction

sample.to_csv('submission.csv', index=False)



print(sample.head(20))



iii,ii = np.unique(sample['diagnosis'], return_counts=True)

print('\n',ii)
# serialize model to JSON

model_json = model_resnet.to_json()

with open("model.json", "w") as json_file:

    json_file.write(model_json)

    

# serialize weights to HDF5

model_resnet.save_weights("model.h5")

print("Saved model to disk")