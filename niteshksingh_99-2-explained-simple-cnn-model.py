import os

import numpy as np

import pandas as pd

import cv2

import matplotlib.pyplot as plt

import tensorflow as tf

import random

from sklearn.model_selection import train_test_split

from PIL import Image,ImageFilter
df = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

X_test_main = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
df.head()
Y = df.iloc[:,0]

X = df.iloc[:,1:]
X = X.to_numpy()

Y = Y.to_numpy()

X_test_main = X_test_main.to_numpy()
X = X.reshape(-1,28,28)

X_test_main = X_test_main.reshape(-1,28,28)

print(X.shape)

print(X_test_main.shape)
w=14

h=14

fig=plt.figure(figsize=(w,h))

columns = 4

rows = 5

for i in range(1, rows*columns+1):

    img1 = X[i+random.randrange(1,300)]

    fig.add_subplot(rows, columns, i)

    plt.imshow(img1)

plt.show()
X = np.expand_dims(X,axis=-1)

X_test_main = np.expand_dims(X_test_main,axis=-1)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size= 0.2,shuffle=True)
print(X_train.shape,Y_train.shape)

print(X_test.shape,Y_test.shape)

print(X_test_main.shape)
X_train = X_train/255.

X_test = X_test/255.

X_test_main = X_test_main/255.

print(X_train.shape)

print(X_test.shape)

print(X_test_main.shape)
def one_hottie(labels,C):

    """

    One hot Encoding is used in multi-class classification problems to encode every label as a vector of binary values

        eg. if there are 3 class as 0,1,2

            one hot vector for class 0 could be : [1,0,0]

                           then class 1: [0,1,0]

                           and class 2: [0,0,1]

    We need this encoding in out labels for the model learns to predict in a similar way.

    

    Without it,if only integer values are used in labels,it could affect model in different ways,

        such as predicting a class that does not exist.

        

    """

    One_hot_matrix = tf.one_hot(labels,C)

    return tf.keras.backend.eval(One_hot_matrix)



Y_test_later = Y_test.copy()

Y_train = one_hottie(Y_train, 10)

Y_test = one_hottie(Y_test, 10)

print ("Y shape: " + str(Y_train.shape))

print ("Y test shape: " + str(Y_test.shape))
# def res_net_block(input_data, filters=[128], conv_size=[3,5]):

#     x = tf.keras.layers.Conv2D(filters[0], conv_size[0], activation='relu', padding='same')(input_data)

#     x = tf.keras.layers.BatchNormalization()(x)

#     x = tf.keras.layers.Conv2D(filters[0], conv_size[1], activation=None, padding='same')(x)

#     x = tf.keras.layers.BatchNormalization()(x)

#     x = tf.keras.layers.Add()([x, input_data])

#     x = tf.keras.layers.Activation('relu')(x)

#     return x
model = tf.keras.Sequential([

    tf.keras.layers.Conv2D(64, 3, activation='relu', input_shape=(28,28,1),padding="same"),

    tf.keras.layers.MaxPool2D(strides=2),

    

    

    tf.keras.layers.Conv2D(128, 3, activation='relu',padding="same"),

    tf.keras.layers.MaxPool2D(strides=2),

    

    tf.keras.layers.Dropout(0.2),

        

    tf.keras.layers.Conv2D(256, 3, activation='relu',padding="same"),

    tf.keras.layers.MaxPool2D(strides=2),

    

    tf.keras.layers.Conv2D(256, 3, activation='relu',padding="same"),

    tf.keras.layers.MaxPool2D(strides=2),

        

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(100,kernel_regularizer=tf.keras.regularizers.l2(0.01), activation='relu'),

    

    tf.keras.layers.Dense(50,kernel_regularizer=tf.keras.regularizers.l2(0.01), activation='relu'),

        

    tf.keras.layers.Dense(10, kernel_regularizer=tf.keras.regularizers.l2(0.01) ,activation='softmax')

])
model.summary()
# initial_learning_rate = 0.001 #initial rate

# # Rate decay with exponential decay

# # new rate = initial_learning_rate * decay_rate ^ (step / decay_steps)



# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(

#     initial_learning_rate,

#     decay_steps=800,

#     decay_rate=0.5,

#     staircase=True)
model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=0.006),

              loss=tf.keras.losses.CategoricalCrossentropy(),

              metrics=['accuracy','Recall','Precision'])



result = model.fit(x=X_train,y=Y_train,batch_size=64,epochs=20,verbose=1,shuffle=False,initial_epoch=0,

                   validation_split=0.1)
plt.plot(result.history['acc'], label='train')

plt.plot(result.history['val_acc'], label='valid')

plt.legend(loc='upper left')

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.show()

plt.plot(result.history['loss'], label='train')

plt.plot(result.history['val_loss'], label='test')

plt.legend(loc='upper right')

plt.title('Model Cost')

plt.ylabel('Cost')

plt.xlabel('Epoch')

plt.show()
model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=0.0001),

              loss=tf.keras.losses.CategoricalCrossentropy(),

              metrics=['accuracy','Recall','Precision'])



result = model.fit(x=X_train,y=Y_train,batch_size=64,epochs=40,verbose=1,shuffle=False,initial_epoch=20,

                   validation_split=0.1)
plt.plot(result.history['acc'], label='train')

plt.plot(result.history['val_acc'], label='valid')

plt.legend(loc='upper left')

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.show()

plt.plot(result.history['loss'], label='train')

plt.plot(result.history['val_loss'], label='test')

plt.legend(loc='upper right')

plt.title('Model Cost')

plt.ylabel('Cost')

plt.xlabel('Epoch')

plt.show()
model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=0.00006),

              loss=tf.keras.losses.CategoricalCrossentropy(),

              metrics=['accuracy','Recall','Precision'])



result = model.fit(x=X_train,y=Y_train,batch_size=64,epochs=60,verbose=1,shuffle=False,initial_epoch=40,

                   validation_split=0.1)
check = model.evaluate(X_test,Y_test)
preds = model.predict_classes(X)

preds.shape
# X = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

# Y_test = X.iloc[:,0]

# Y_test = Y_test.to_numpy()
conf = tf.math.confusion_matrix(preds,Y)
with tf.Session() as session:

    print(conf.eval())
train_gen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=20,

                                                            zoom_range=0.20,

                                                            width_shift_range=0.2,

                                                            height_shift_range=0.2,

                                                            shear_range=0.20,

                                                            horizontal_flip=False,

                                                            brightness_range=[0.1,1],

                                                            rescale=1./255)

test_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=0.00005),

              loss=tf.keras.losses.CategoricalCrossentropy(),

              metrics=['accuracy','Recall','Precision'])



result = model.fit(train_gen.flow(X_train*255,Y_train,batch_size=64),

                   validation_data = test_gen.flow(X_test*255,Y_test,batch_size=16),

                   epochs=70,

                   verbose=1)
preds = model.predict_classes(X_test_main)
preds.shape
arr = [x for x in range(1,28001)]

label = pd.DataFrame(arr,columns = ["ImageId"])

label["Label"] = pd.DataFrame(preds)

label.head()
label.to_csv('Y_test.csv',header=True,index = False)
model.save("saved_model")