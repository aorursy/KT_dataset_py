import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from glob import glob

from PIL import Image



from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split





import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten

from tensorflow.keras.optimizers import Adam





from keras.utils import to_categorical



import tensorflow.keras.backend as K
class_names = glob('../input/*/train/*')

for i in range (len(class_names)):

    class_names[i] = class_names[i][len('../input/logos3/train/'):]

class_names



class_ind = {class_names[i]:i for i in range(6)}

print(class_names)

print(class_ind)
image_names = {}

count_list = {}

for class_name in class_names:

    image_names[class_name] = glob('../input/*/*/' + class_name +  '/*' )

    count_list[class_name] = len(glob('../input/*/*/' + class_name +  '/*' ))

#     print(class_name,'-' , len(image_names[class_name]),'images')

count_list
dataset = pd.DataFrame(list(count_list.items()),columns=['name','number'])

dataset
import random

r = lambda: random.randint(0,255)

col = lambda : ('#%02X%02X%02X' % (r(),r(),r()))
import seaborn as sns

sns.set(style="darkgrid")

ax = plt.bar(dataset['name'],dataset['number'] , color = [col() for i in range(6)])

plt.show()
sns.set_style("whitegrid", {'axes.grid' : False})
j = 0

fig, axs = plt.subplots(6, 5, figsize=(20, 30))

for class_name in class_names:

    class_wise_names = image_names[class_name]

    i = 0

    for img_name in class_wise_names[:5]:

        img = Image.open(img_name)

        img = np.array(img)

        axs[j][i].imshow(img)

        axs[j][i].set_title(class_name)

        i+=1

    j+=1
# def generate_image_arr(cur_dir,image_size = (224,224)):

#     image_names = {}

#     count_list = {}

#     m = 0

#     for class_name in class_names:

#         image_names[class_name] = glob(cur_dir + class_name +  '/*' )

#         count_list[class_name] = len(image_names[class_name])

#         m += len(image_names[class_name])

#     print(count_list)

#     print('Total image count =', m)

    

#     X, Y = np.zeros((m,*image_size,3),dtype=np.uint8),np.zeros((m,1),dtype=np.uint8)

    

#     ind = 0 

#     for class_name in class_names:

#         class_wise_names = image_names[class_name]

#         for i in range(len(class_wise_names)):

#             img = Image.open(class_wise_names[i])

#             img = img.resize(image_size)

#             img = np.array(img)

#             X[ind]= img

#             Y[ind] = class_ind[class_name]

#             assert(class_name == class_names[class_ind[class_name]])

#             ind+=1

#     Y_ = to_categorical(Y)

# #     print(Y_.shape)

# #     print(Y.shape)

# #     print(.shape)

# #     assert(np.argmax(Y_,axis=1).reshape(-1,1) == Y)

# #     assert(np.argmax(Y_,axis=1) == Y.flatten())

#     assert(np.argmax(Y_,axis = 1).reshape(-1,1).shape == Y.shape)

    

    

#     return X,Y_,m
# cur_dir = '../input/*/train/'

# X_, Y_, m_tr = generate_image_arr(cur_dir)

# Xtr, Xval, Ytr, Yval = train_test_split(X_,Y_,test_size=0.25)

# print(Xtr.shape)

# print(Xval.shape)

# print(Ytr.shape)

# print(Yval.shape)
# fig, axs = plt.subplots(5, 5, figsize=(20, 30))

# for i in range(25):

#     ax = axs[i//5][i % 5]

#     ax.imshow(Xtr[i])

#     ax.set_title(class_names[np.argmax(Ytr[i])] +" " +str(Ytr[i]))
# cur_dir = '../input/*/test/'

# Xts, Yts, m_ts  = generate_image_arr(cur_dir)


# fig, axs = plt.subplots(5, 5, figsize=(20, 30))

# for i in range(25):

#     ax = axs[i//5][i % 5]

#     i = random.randint(0,560)

#     ax.imshow(Xts[i])

#     ax.set_title(class_names[np.argmax(Yts[i])] +" " +str(Yts[i]))
# def get_images_of_class(class_name):

# #     print(Xts[(np.argmax(Yts,axis=-1).reshape(-1,1) == [class_ind[class_name]]).flatten()].shape)

#     return Xts[(np.argmax(Yts,axis=-1).reshape(-1,1) == [class_ind[class_name]]).flatten()]

    

# fig, axs = plt.subplots(2, 3, figsize=(20, 15))

# axs = axs.flatten()

# for i in range(len(class_names)):

#     mean_img = np.array(get_images_of_class(class_names[i])[0:5].mean(axis=0),dtype=np.uint8)

#     ax = axs[i]

#     ax.imshow(mean_img)

#     ax.set_title(class_names[i])

# #     print(mean_img.shape)

# #     plt.imshow(mean_img)

# #     print(ax)

# # plt.imshow(mean_img)
# model = Sequential()

# model.add(Conv2D(input_shape=(224,224,3),filters=32,kernel_size=(3,3),padding="same", activation="relu"))

# model.add(Conv2D(filters=32,kernel_size=(3,3),padding="same", activation="relu"))

# model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))



# model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))

# model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))

# model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))



# model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))

# model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))

# model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))

# model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))



# model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))

# model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))

# model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))

# model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))





# model.add(Flatten())

# model.add(Dense(units=1024,activation="relu"))

# model.add(Dense(units=1024,activation="relu"))

# model.add(Dense(units=6, activation="softmax"))
# model.summary()


# opt = Adam(lr=0.001)

# model.compile(optimizer=opt, loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy', 'acc'])
# # del X_,Y_

# import gc

# gc.collect()
# datagen = tf.keras.preprocessing.image.ImageDataGenerator(

#     featurewise_center=True,

#     featurewise_std_normalization=True,

#     rotation_range=20,

#     width_shift_range=0.2,

#     height_shift_range=0.2,

#     horizontal_flip=True

# )





# # train_generator = datagen.flow_from_directory(

# #     directory=r"../input/logos3/train/",

# #     target_size=(224, 224),

# #     color_mode="rgb",

# #     batch_size=32,

# #     class_mode="categorical",

# #     shuffle=True,

# #     seed=42

# # )
# datagen.fit(Xtr[:1],seed=7)
# hist = model.fit(datagen.flow(Xtr, Ytr, batch_size=32),

#           steps_per_epoch=len(Xtr) / 32,validation_data = datagen.flow(Xval,Yval), validation_steps=10, epochs=40,seed=7)
# 
# import ctypes

# # a = "hello world"

# myid = int('0x7efe6ba1ff60',16)

# print(ctypes.cast(myid, ctypes.py_object).value)

# import matplotlib.pyplot as plt



# plt.plot(hist.history["acc"])

# plt.plot(hist.history['val_acc'])

# plt.title("model accuracy")

# plt.ylabel("Accuracy")

# plt.xlabel("Epoch")

# plt.legend(["Accuracy","Validation Accuracy"])

# plt.show()
# # pred = np.argmax(model.predict(Xtr),axis=1)



# test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)

# test_datagen.fit(Xts[:1])



# pred = model.predict(test_datagen.flow(Xts,shuffle=0,batch_size=1),steps = Xts.shape[0])

# ypr = np.argmax(pred,axis=1)

# y = np.argmax(Yts,axis=1)

# acc = sum(ypr == y)/Xts.shape[0]

# acc





# predictions = model.predict(x=testX.astype("float32"), batch_size=BS)

# print(classification_report(testY.argmax(axis=1),

# 	predictions.argmax(axis=1), target_names=le.classes_))
# model.save('my_model.h5')
train_dir = glob('../input/*/train')[0]

test_dir = glob('../input/*/test')[0]
datagen = tf.keras.preprocessing.image.ImageDataGenerator(

    featurewise_center=True,

    featurewise_std_normalization=True,

    rotation_range=20,

    width_shift_range=0.2,

    height_shift_range=0.2,

    horizontal_flip=True,

    validation_split=0.2

)



datagen.fit(img.reshape(1,300,300,3)[:,0:224,0:224,:])



# train_flowed = train_gen.flow_from_directory(

#     train_dir, target_size=(224, 224), color_mode='rgb', classes=None,

#     class_mode='categorical', batch_size=32, shuffle=True, seed=724

# )



train_gen = datagen.flow_from_directory(

    train_dir,

    target_size=(224, 224),

    batch_size=32,

    class_mode='categorical',

    subset='training') # set as training data



val_gen = datagen.flow_from_directory(

    train_dir,

    target_size=(224, 224),

    batch_size=32,

    class_mode='categorical',

    subset='validation') # set as validation data



test_gen = datagen.flow_from_directory(

    test_dir,

    target_size=(224, 224),

    batch_size=32,

    class_mode='categorical')
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D

from tensorflow.keras.layers import BatchNormalization

import numpy as np

np.random.seed(1000)

#Instantiate an empty model

model = Sequential()



# 1st Convolutional Layer

model.add(Conv2D(filters=96, input_shape=(224,224,3), kernel_size=(11,11), strides=(4,4), padding='valid',activation="relu"))

model.add(Activation('relu'))

# Max Pooling

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))



# 2nd Convolutional Layer

model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))

model.add(Activation('relu'))

# Max Pooling

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))



# 3rd Convolutional Layer

model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))

model.add(Activation('relu'))



# 4th Convolutional Layer

model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))

model.add(Activation('relu'))



# 5th Convolutional Layer

model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))

model.add(Activation('relu'))

# Max Pooling

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))



# Passing it to a Fully Connected layer

model.add(Flatten())

# 1st Fully Connected Layer

model.add(Dense(4096, input_shape=(224*224*3,)))

model.add(Activation('relu'))

# Add Dropout to prevent overfitting

model.add(Dropout(0.4))



# 2nd Fully Connected Layer

model.add(Dense(4096))

model.add(Activation('relu'))

# Add Dropout

model.add(Dropout(0.4))



# 3rd Fully Connected Layer

model.add(Dense(1000))

model.add(Activation('relu'))

# Add Dropout

model.add(Dropout(0.4))



# Output Layer

model.add(Dense(6))

model.add(Activation('softmax'))



model.summary()



# Compile the model



def f1_score(y_true, y_pred): #taken from old keras source code

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    precision = true_positives / (predicted_positives + K.epsilon())

    recall = true_positives / (possible_positives + K.epsilon())

    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())

    return f1_val
opt = Adam(lr=0.001)

model.compile(optimizer=opt, loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy', 'acc', f1_score])
hist = model.fit_generator(train_gen,

          steps_per_epoch= train_gen.samples // 32, validation_data = val_gen, validation_steps=val_gen.samples // 32, epochs=40 )
import matplotlib.pyplot as plt



plt.plot(hist.history["acc"])

plt.plot(hist.history['val_acc'])

plt.title("model accuracy")

plt.ylabel("Accuracy")

plt.xlabel("Epoch")

plt.legend(["Accuracy","Validation Accuracy"])

plt.show()


# filenames = test_gen.filenames

# nb_samples = len(filenames)
_, acc, f1_score_val = model.evaluate(test_gen)
acc, f1_score_val
model = Sequential()

model.add(Conv2D(input_shape=(224,224,3),filters=32,kernel_size=(3,3),padding="same", activation="relu"))

model.add(Conv2D(filters=32,kernel_size=(3,3),padding="same", activation="relu"))

model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))

model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))

model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))

model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))

model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))

model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))

model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))

model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))

model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))

model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))

model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))

model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))





model.add(Flatten())

model.add(Dense(units=1024,activation="relu"))

model.add(Dense(units=1024,activation="relu"))

model.add(Dense(units=6, activation="softmax"))
model.summary()
opt = Adam()

model.compile(loss = tf.keras.losses.categorical_crossentropy, optimizer='adam', metrics=["accuracy", f1_score])
hist = model.fit_generator(train_gen,

          steps_per_epoch= train_gen.samples // 32, validation_data = val_gen, validation_steps=val_gen.samples // 32, epochs=40 )
_, acc, f1_score_val = model.evaluate(test_gen)
acc, f1_score_val