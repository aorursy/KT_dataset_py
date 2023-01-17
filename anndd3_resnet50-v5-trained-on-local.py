import numpy as np

import pandas as pd

from tqdm import tqdm_notebook

from zipfile import ZipFile

import os

import cv2

from matplotlib import cm

import matplotlib.pyplot as plt



from keras.models import Sequential

from keras_preprocessing.image import ImageDataGenerator

from keras.layers import Activation, Dropout, Conv2D, GlobalMaxPooling2D, MaxPooling2D, Flatten, Dense, BatchNormalization

from keras.optimizers import Adam, SGD

from sklearn.utils import compute_class_weight

from keras.applications.resnet50 import ResNet50

from keras.regularizers import l2



from keras_tqdm import TQDMNotebookCallback

from keras.callbacks import ModelCheckpoint
#! ls ../input
#!rm -rf ../input/trainedprocessed
base_path = "../input/"

trainProcessed = "trainedprocessed"

aptop = 'aptos2019-blindness-detection/'

localModel = 'trained-on-local/'

train = "train_images"

IMG_DIM = 96

SEED = 6819
data_df = pd.read_csv("../input/aptos2019-blindness-detection/train.csv")
data_df['id_code'] = data_df['id_code']+".png" 

data_df['diagnosis'] = data_df['diagnosis'].astype("str")
data_df.head()
def crop_image_from_gray_sample(img,tol=7):

    """

    Crop out black borders

    https://www.kaggle.com/ratthachat/aptos-updated-preprocessing-ben-s-cropping

    """ 

    if img.ndim ==2:

        mask = img>tol

        return img[np.ix_(mask.any(1),mask.any(0))]

    elif img.ndim==3:

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        mask = gray_img>tol        

        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]

        if (check_shape == 0):

            return img

        else:

            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]

            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]

            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]

            img = np.stack([img1,img2,img3],axis=-1)

        return img
imgs = cv2.imread(base_path+aptop+"train_images/000c1434d8d7.png")

imgs = crop_image_from_gray_sample(imgs)

imgs = cv2.resize(imgs, (IMG_DIM, IMG_DIM))

imgs = cv2.addWeighted(imgs,4, cv2.GaussianBlur(imgs, (0,0), 10), -4, 128)
#Experiment

plt.imshow(imgs)
def crop_image_from_gray(img,tol=7):

    """

    Crop out black borders

    https://www.kaggle.com/ratthachat/aptos-updated-preprocessing-ben-s-cropping

    """  

    if img.ndim ==2:

        mask = img>tol

        return img[np.ix_(mask.any(1),mask.any(0))]

    elif img.ndim==3:

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        mask = gray_img>tol        

        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]

        if (check_shape == 0):

            return img

        else:

            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]

            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]

            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]

            img = np.stack([img1,img2,img3],axis=-1)

        return img
def pre_process(img):

    img = crop_image_from_gray((img).astype(np.uint8))

    img = cv2.resize(img, (IMG_DIM, IMG_DIM))

    img = cv2.addWeighted(img,4, cv2.GaussianBlur(img, (0,0), 10), -4, 128)

    return img.astype(np.float64)
try:

    os.mkdir(base_path+trainProcessed)

    print("Directory Created :"+trainProcessed)

    

    for file in tqdm_notebook(data_df["id_code"].iteritems(), total=data_df.shape[0]):

        img = cv2.imread(base_path+aptop+train+"/"+file[1])

        img = crop_image_from_gray(img)

        img = cv2.resize(img, (400, 400))

        img = cv2.addWeighted(img,4, cv2.GaussianBlur(img, (0,0), 10), -4, 128)

        cv2.imwrite(base_path+trainProcessed+"/"+file[1], img)

    print("Conversion done.")

except:

    print("Directory "+trainProcessed+" already exist !")
batch_size=8

epochs = 20
data_gen = ImageDataGenerator(vertical_flip=True,

                             horizontal_flip=True,

                             rescale=1./255, 

                             #preprocessing_function=pre_process,

                             data_format='channels_last')
tempGen = data_gen.flow_from_dataframe(dataframe=data_df, 

                                       directory=base_path+trainProcessed,

                                       x_col="id_code",

                                       y_col="diagnosis",

                                       target_size=(IMG_DIM,IMG_DIM),

                                       class_mode='categorical',

                                       batch_size=batch_size,

                                       seed=SEED)
tx, ty = next(tempGen)
fig = plt.figure(figsize=(100, 100))

columns = 4

row = 2

for i in range(1, columns*row+1):

    fig.add_subplot(row, columns, i)

    #plt.imshow((tx[i-1].reshape((400, 400))*255).astype(np.int32))

    plt.imshow((tx[i-1]*255).astype(np.uint8))

plt.show()
train_df = data_df.sample(frac=0.8, random_state=SEED)

validate_df = data_df.loc[~data_df['id_code'].isin(train_df['id_code'].values)]
len(train_df), len(validate_df)
train_df.reset_index(inplace=True, drop=True)

validate_df.reset_index(inplace=True, drop=True)
train_generator = data_gen.flow_from_dataframe(dataframe=train_df, 

                                       directory=base_path+trainProcessed,

                                       x_col="id_code",

                                       y_col="diagnosis",

                                       target_size=(IMG_DIM,IMG_DIM),

                                       class_mode='categorical',

                                       batch_size=batch_size,

                                       seed=SEED)

validation_generator = data_gen.flow_from_dataframe(dataframe=validate_df, 

                                       directory=base_path+trainProcessed,

                                       x_col="id_code",

                                       y_col="diagnosis",

                                       target_size=(IMG_DIM,IMG_DIM),

                                       class_mode='categorical',

                                       batch_size=batch_size,

                                       seed=SEED)
X, y = next(train_generator)
X.shape, y.shape
class_weights = compute_class_weight("balanced", 

                     np.unique(train_generator.classes), 

                     train_generator.classes)

class_weights
model_resnet = ResNet50(include_top=False, 

                        weights = None,

                        input_shape=(IMG_DIM, IMG_DIM, 3,))

model_resnet.summary()
model = Sequential()

model.add(model_resnet)



#model.add(Flatten())

model.add(GlobalMaxPooling2D())

model.add(Dense(1000, activation='relu', activity_regularizer=l2(0.001)))

model.add(Dense(500, activation='relu', activity_regularizer=l2(0.001)))

#model.add(Dense(250, activation='relu', activity_regularizer=l2(0.001)))

#model.add(BatchNormalization())

#model.add(Dropout(0.5))

model.add(Dense(5, activation = 'softmax'))
model.summary()
model.load_weights(base_path+localModel+"model_resnet_aptos_0.2.hdf5")
model.compile(loss='categorical_crossentropy',

              #optimizer=SGD(lr=0.00001),

              optimizer=Adam(lr=0.00001),

              metrics=['accuracy'])
filepath="weights-resnet-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"

#filepath="weights.best.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')

callbacks_list = [checkpoint, TQDMNotebookCallback()]

#callbacks_list = [TQDMNotebookCallback()]
train_step = train_df.size//batch_size

valid_step = validate_df.size//batch_size

history = model.fit_generator(train_generator,

                    validation_data=validation_generator,

                    epochs=epochs, 

                    steps_per_epoch=train_step, 

                    validation_steps=valid_step, 

                    shuffle= True,

                    workers=6, 

                    verbose=2,

                    use_multiprocessing=False,

                    #callbacks=callbacks_list, 

                    initial_epoch=0,

                    class_weight=class_weights)
model.save('model_resnet_aptos_kaggle.hdf5')
test = "test_images/"

testProcessed = "testProcessed"
test_df = pd.read_csv(base_path+aptop+"test.csv")
test_df['id_code'] = test_df['id_code']+".png"
test_df.head()
#!ls ../input

#!rm -rf  ../input/testProcessed
try:

    os.mkdir(base_path+testProcessed)

    print("Directory Created :"+testProcessed)

    

    for file in tqdm_notebook(test_df["id_code"].iteritems(), total=test_df.shape[0]):

        img = cv2.imread(base_path+aptop+test+"/"+file[1])

        img = crop_image_from_gray(img)

        img = cv2.resize(img, (IMG_DIM, IMG_DIM))

        img = cv2.addWeighted(img,4, cv2.GaussianBlur(img, (0,0), 10), -4, 128)

        cv2.imwrite(base_path+testProcessed+"/"+file[1], img)

    print("Conversion done.")

except:

    print("Directory "+testProcessed+" already exist !")
test_datagen = ImageDataGenerator(rescale=1./255, 

                                  #preprocessing_function=pre_process,

                                  data_format="channels_last")
testgen = test_datagen.flow_from_dataframe(dataframe=test_df,

                                       directory=base_path+aptop+test, 

                                       target_size=(IMG_DIM,IMG_DIM), 

                                       x_col="id_code", 

                                       batch_size=batch_size, 

                                       shuffle=False, 

                                       class_mode=None, 

                                       seed=SEED)
testgen.reset()

STEP_TEST_GEN = test_df.size//batch_size

pred = model.predict_generator(testgen,

                               steps=STEP_TEST_GEN,

                               verbose=1)
pred_class_indices = np.argmax(pred, axis=1)
labels = (train_generator.class_indices)
labels
labels = (train_generator.class_indices)

labels = dict((c,v) for v, c in labels.items())

prediction = [labels[k] for k in pred_class_indices]
filenames = testgen.filenames
results = pd.DataFrame({"id_code": filenames, 

          "diagnosis": prediction})
results.head()
results["id_code"] = results["id_code"].str.rstrip(".png")
results.to_csv("submission.csv", index=False)