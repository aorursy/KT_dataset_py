import os

import cv2

import random

import shutil

import numpy as np 

import pandas as pd 

import seaborn as sns

from keras.models import load_model

import matplotlib.pyplot as plt

from matplotlib.image import imread

from sklearn.metrics import confusion_matrix,classification_report

from keras.preprocessing.image import ImageDataGenerator,load_img

from keras.models import Sequential,Model

from keras.layers import Conv2D,Flatten,MaxPooling2D,Dense,GlobalAveragePooling2D,Dropout,BatchNormalization

from sklearn.model_selection import train_test_split

from keras.callbacks import ReduceLROnPlateau
import zipfile

with zipfile.ZipFile("../input/dogs-vs-cats/"+'train'+".zip","r") as z:

    z.extractall(".")
Y = []

path = "./train"

filenames = os.listdir(path)

for img in os.listdir(path):

    val = img.split(".")[0]

    if val == "dog":

        Y.append('1')

    else:

        Y.append('0')



df = pd.DataFrame({

    'filename' : filenames,

    'category' : Y

})
df.head()
df.tail()
plt.figure(figsize=(12, 12))

for i in range(0, 9):

    plt.subplot(4, 3, i+1)

    sample = random.choice(filenames)

    filename = path+'/'+sample

    image = imread(filename)

    plt.imshow(image)

plt.tight_layout()

plt.show()
train_df,val_df = train_test_split(df,test_size=0.2,random_state = 42)

train_df = train_df.reset_index(drop=True)

val_df = val_df.reset_index(drop=True)
train_df.shape
val_df.shape
train_df['category'].value_counts().plot.bar()
val_df['category'].value_counts().plot.bar()
batch_size = 32

epochs = 30

train_size = train_df.shape[0]

val_size = val_df.shape[0]

img_hieght = 128

img_width = 128

img_channels = 3
train_datagen = ImageDataGenerator(

    rescale = 1./255,

    rotation_range = 15,

    horizontal_flip = True,

    zoom_range = 0.2,

    shear_range = 0.1,

    fill_mode = 'reflect',

    width_shift_range = 0.1,

    height_shift_range = 0.1

)



train_generator = train_datagen.flow_from_dataframe(

    train_df,

    "./train",

    x_col = 'filename',

    y_col = 'category',

    target_size = (img_hieght,img_width),

    batch_size = batch_size,

    class_mode = 'binary'

)
example_df = train_df.sample(n=1)

example_generator = train_datagen.flow_from_dataframe(

    example_df, 

    "./train",

    x_col='filename',

    y_col='category',

    target_size=(img_hieght,img_width),

    class_mode='raw'

)
plt.figure(figsize=(12, 12))

for i in range(0, 9):

    plt.subplot(4, 3, i+1)

    for X_batch, Y_batch in example_generator:

        image = X_batch[0]

        plt.imshow(image)

        break

plt.tight_layout()

plt.show()
val_datagen = ImageDataGenerator(

    rescale = 1./255,

)



val_generator = val_datagen.flow_from_dataframe(

    val_df,

    "./train",

    x_col = 'filename',

    y_col = 'category',

    target_size = (img_hieght,img_width),

    batch_size = batch_size,

    class_mode = 'binary'

)
model = Sequential()



model.add(Conv2D(32,(3,3),activation='relu',input_shape = (img_hieght,img_width,img_channels)))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(64,(3,3),activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(128,(3,3),activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(256,(3,3),activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(512,activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.25))

model.add(Dense(1,activation='sigmoid'))
learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_accuracy',

                                            patience=2,

                                            factor=0.5,

                                            min_lr = 0.00001,

                                            verbose = 1)

callbacks = [learning_rate_reduction]
model.summary()
model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit_generator(

    train_generator,

    epochs = epochs,

    validation_data =  val_generator,

    steps_per_epoch = train_size//batch_size,

    validation_steps = val_size//batch_size,

    callbacks = callbacks

)
model.save("model.h5")
# model = load_model("/kaggle/input/mode-file/model.h5")
score = model.evaluate_generator(val_generator)

print(score[1]*100)
score = model.evaluate_generator(train_generator)

print(score[1]*100)
conf_datagen = ImageDataGenerator(

    rescale = 1./255,

)



conf_generator = conf_datagen.flow_from_dataframe(

    val_df,

    "./train",

    x_col = 'filename',

    y_col = 'category',

    target_size = (img_hieght,img_width),

    batch_size = batch_size,

    shuffle = False,

    class_mode = 'binary'

)
y_predict = model.predict_generator(conf_generator)
y_predict = np.where(y_predict > 0.5, 1, 0)
p = conf_generator.classes

q = y_predict

p = np.array(p)

q = q.flatten()
cfm = confusion_matrix(p, q)

print(cfm)

ax= plt.subplot()

sns.heatmap(cfm, annot=True, ax = ax);

# labels, title and ticks

ax.set_xlabel('Predicted labels')

ax.set_ylabel('True labels');

ax.set_title('Confusion Matrix');

ax.xaxis.set_ticklabels(['cats', 'dogs'])

ax.yaxis.set_ticklabels(['cats', 'dogs'])
print(classification_report(p,q))
path = "/kaggle/input/test-images/dog.jpeg"

img = cv2.imread(path)

plt.imshow(img)

img = cv2.resize(img,(128,128))

img = np.reshape(img,[1,128,128,3])

img = np.divide(img,255)

result = model.predict(img)

if result[0] >= 0.5:

    print("According to our model's prediction below image is of a Dog")

else:

    print("According to our model's prediction below image is of a Cat")
shutil.rmtree("./train")
# from keras.applications.resnet50 import ResNet50
# base_model = ResNet50(include_top=False,weights=None,input_shape=(128,128,3))

# res_model = base_model.output

# res_model = GlobalAveragePooling2D()(res_model)

# res_model = Dropout(0.5)(res_model)

# predictions = Dense(1,activation='sigmoid')(res_model)

# model = Model(inputs = base_model.input,outputs = predictions)