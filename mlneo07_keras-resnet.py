from keras.preprocessing.image import ImageDataGenerator

from keras import applications

from keras.models import Model

from keras.layers import Dense, Flatten, Dropout,GlobalMaxPooling2D

import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))

import cv2

from tqdm import tqdm_notebook

import matplotlib.pyplot as plt



%matplotlib inline

train_df = pd.read_csv("../input/aptos2019-blindness-detection/train.csv")

test_df = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')



train_dir = '../input/aptos2019-blindness-detection/train_images'

test_dir = '../input/aptos2019-blindness-detection/test_images'
train_df["id_code"]=train_df["id_code"].apply(lambda x:x+".png")

train_df['diagnosis'] = train_df['diagnosis'].astype(str)



test_df["id_code"]=test_df["id_code"].apply(lambda x:x+".png")
train_df.head()
test_df.head()
imgs = []

for imgk in tqdm_notebook(os.listdir(train_dir)[:5]):

        path = os.path.join(train_dir,imgk)

        imgl = cv2.imread(path,cv2.IMREAD_COLOR)

        imgs.append(np.array(imgl))
imgs = np.asarray(imgs)

print(imgs[1].shape)
   

fig, axs = plt.subplots(2, 2)

axs[0, 0].imshow(imgs[0])

axs[1, 0].imshow(imgs[1])

axs[0, 1].imshow(imgs[2])

axs[1, 1].imshow(imgs[3])



plt.show()
imgsize = 250
model = applications.ResNet50(include_top=False,input_shape= (imgsize,imgsize,3), 

                              weights = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
#model.trainable = False

for layer in model.layers[:-15]:

       layer.trainable = False
model_x = model.output

model_x = GlobalMaxPooling2D()(model_x)

model_x = Dropout(0.3)(model_x)

model_x = Dense(1024,activation='relu')(model_x)

model_x = Dropout(0.3)(model_x)

predictions = Dense(5, activation='softmax')(model_x)



model_output = Model(inputs=model.input, outputs=predictions)
X_tr = [] #training image

Y_tr = [] #training lables

test_img = [] #test images
#loading training images and labels

def load_train_image(f_path):

        imges = train_df['id_code']

        for img_id in tqdm_notebook(imges):

                img = cv2.imread(os.path.join(f_path, img_id), cv2.IMREAD_COLOR)

                img = cv2.resize(img,(imgsize,imgsize))

                X_tr.append(np.array(img))

                Y_tr.append(train_df[train_df['id_code'] == img_id]['diagnosis'].values[0])  

        return X_tr, Y_tr
#loading test images

def load_test_image(ft_path):

        imgts = test_df['id_code']

        for img_id_ts in tqdm_notebook(imgts):

                imgk = cv2.imread(os.path.join(ft_path, img_id_ts), cv2.IMREAD_COLOR)

                imgk = cv2.resize(imgk,(imgsize,imgsize))

                test_img.append(np.array(imgk))

                  

        return test_img
load_train_image(f'../input/aptos2019-blindness-detection/train_images/')



X_tr = np.array(X_tr)

X_tr = X_tr.astype('float32')

X_tr /= 255

Y_tr = np.array(Y_tr)
load_test_image(f'../input/aptos2019-blindness-detection/test_images/')



test_img = np.array(test_img)

test_img = test_img.astype('float32')

test_img /= 255
%%time

data_gen = ImageDataGenerator(horizontal_flip = True,

                              vertical_flip = True,

                             rotation_range=20,

                             width_shift_range=0.2,

                             height_shift_range=0.2,

                             validation_split = 0.20

                                                   )

train_aug = data_gen.flow(X_tr,Y_tr,

                          subset = 'training')

train_valid_aug = data_gen.flow(X_tr,

                               Y_tr,

                               subset = 'validation')



test_aug = data_gen.flow(test_img)
model_output.compile(loss='sparse_categorical_crossentropy',             

              optimizer='adam',      

              metrics=['acc'])
history = model_output.fit_generator(generator = train_aug,

                              steps_per_epoch=90,

                              epochs=10,

                              validation_data=train_valid_aug,

                              validation_steps=20

                              )
# summarize history for accuracy

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
batch_size = 32

result = model_output.predict_generator(test_aug ,steps = (test_img.shape[0] // batch_size)+1)
print(result)