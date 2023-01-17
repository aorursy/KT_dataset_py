import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    print(dirname)
# imports used in this project

# keras
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
# ploting
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# the images dirs
train_dir = '/kaggle/input/intel-image-classification/seg_train/seg_train'
test_dir = '/kaggle/input/intel-image-classification/seg_test/seg_test'
pred_dir = '/kaggle/input/intel-image-classification/seg_pred'# specify parent folder for unlabeled images
row = 0
fig = plt.figure(figsize=(9,18))
for dirname, _, filenames in os.walk(train_dir):
    print(dirname)
    if len(filenames)==0:
        continue
    for i in range(3):
        sp = plt.subplot(6,3,row*3+i+1)
        category = dirname.split('/')[-1]
        sp.set_title(category+str(i+1))
        sp.axis('off')
        img_data = mpimg.imread(dirname+'/'+filenames[i])
        plt.imshow(img_data)
    row+=1
    
datagen = ImageDataGenerator(rescale=1./255,# make each pixel 0~1
    rotation_range=20,# rotate the images
    width_shift_range=0.2, # shift the images
    height_shift_range=0.2, # shift the images
    horizontal_flip=True) # flip the images

testgen = ImageDataGenerator(rescale=1./255)# make each pixel 0~1
                             
# training data
train_generator = datagen.flow_from_directory(
        directory=train_dir,
        target_size=(150, 150),# set image size
        color_mode="rgb",# colorful images
        batch_size=32,# 32 a time
        class_mode="categorical",# categorical classification
        shuffle=True,# shuffle the data
        seed=7 # lucky number
        )
# testing data
test_generator = datagen.flow_from_directory(
        directory=test_dir,
        target_size=(150, 150),# set image size
        color_mode="rgb",# colorful images
        batch_size=32,# 32 a time
        class_mode="categorical",# categorical classification
        shuffle=False,# no need to shuffle when validating
        #seed=7 # lucky number
        )

# predicting data
pred_generator = testgen.flow_from_directory(
        directory=pred_dir,
        target_size=(150, 150),# set image size
        color_mode="rgb",# colorful images
        batch_size=1,# no need to get batch
        class_mode=None,# no class mode
        shuffle=False,# should not shuffle when predicting
        #seed=7 # lucky number
        )
model = keras.models.Sequential([
    # input shape 150*150*3
    # 1st set of layers: Conv2d+BatchNormalization+Relu --> 150*150*3 becomes 150*150*32
    keras.layers.Conv2D(32,(5,5),padding="same",input_shape=(150,150,3)),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    # 2nd set of layers: Conv2d+BatchNormalization+Relu+MaxPooling2D --> 150*150*32 becomes 75*75*32
    keras.layers.Conv2D(32,(3,3),padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.MaxPooling2D(),
    # 3rd set of layers: Conv2d+BatchNormalization+Relu --> 75*75*32 becomes 75*75*64
    keras.layers.Conv2D(64,(3,3),padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    # 4th set of layers: Conv2d+BatchNormalization+Relu+MaxPooling2D --> 75*75*64 becomes 38*38*64
    keras.layers.Conv2D(64,(3,3),padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.MaxPooling2D(padding="same"),
    # 5th set of layers: Conv2d+BatchNormalization+Relu+MaxPooling2D --> 38*38*64 becomes 19*19*128
    keras.layers.Conv2D(128,(3,3),padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.MaxPooling2D(padding="same"),
    # 5th set of layers: Conv2d+BatchNormalization+Relu+MaxPooling2D --> 19*19*128 becomes 10*10*128
    keras.layers.Conv2D(128,(3,3),padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.MaxPooling2D(padding="same"),
    # 5th set of layers: Conv2d+BatchNormalization+Relu+MaxPooling2D --> 10*10*128 becomes 5*5*256
    keras.layers.Conv2D(256,(3,3),padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.MaxPooling2D(padding="same"),
    # 5th set of layers: Conv2d+BatchNormalization+Relu+MaxPooling2D --> 5*5*256 becomes 3*3*256
    keras.layers.Conv2D(256,(3,3),padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.MaxPooling2D(padding="same"),
    # flatten the image --> 3*3*256 becomes 2304
    keras.layers.Flatten(),
    # go through a bunch of neurons and drop some of the links --> 2304 becomes 1024
    keras.layers.Dense(1024, activation="relu"),
    keras.layers.Dropout(0.2),
    # go through another bunch of neurons and drop some of the links --> 1024 becomes 128
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.2),
    # finally go through 6 neurons --> 128 becomes 6
    keras.layers.Dense(6, activation="softmax")
])
# compile model for categorical results
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
#model.summary()
# callback for each epoch
model_path = '/kaggle/working/best_model.h5'
callbacks = [
    # save model
    ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True),
    # stop when changes become little
    EarlyStopping(monitor='val_loss',patience=20)
]
# model.fit_generator(
#     train_generator,
#     epochs=60,
#     validation_data=test_generator,
#     callbacks=callbacks,
#     shuffle=True# shuffle data to get more stable fitting
# )
model.load_weights('/kaggle/input/basic-cnn-for-image-classification-categorical/best_model.h5')
# try predict one image
# anImage = pred_dir+'/seg_pred/10004.jpg'
# imgData = mpimg.imread(anImage)
imgData = pred_generator[1]
# img = np.expand_dims(imgData, axis = 0)
model.predict(imgData)
plt.imshow(imgData[0])
# predict
res = model.predict_generator(pred_generator)
# predicted data looks like this
res[0]
# generate human readable prediction results
cates = ['buildings','forest','glacier','mountain','sea','street']
files = os.listdir(pred_dir+'/seg_pred')
pred_data = {'file':[],'category':[]}
for i in range(len(res)):
    p_category = cates[np.argmax(res[i])]
    p_file = files[i]
    pred_data['file'].append(p_file)
    pred_data['category'].append(p_category)
df = pd.DataFrame(pred_data, columns = ['file', 'category'])

df.to_csv('/kaggle/working/prediction.csv')
df.head()
# select 18 images
img_indexes = np.random.choice(len(df),18)
fig = plt.figure(figsize=(9,18))
for i,img_index in enumerate(img_indexes):
    title = df.iloc[img_index]['category']
    img = pred_generator[img_index][0]
    sp = plt.subplot(6,3,i+1)
    sp.set_title(title)
    sp.axis('off')
    plt.imshow(img)