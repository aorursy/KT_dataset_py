# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os

import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import tensorflow as tf

import random

from PIL import Image,ImageFilter
data=pd.read_csv('../input/dataset-for-test/dataset_00_with_header.csv')

data.head()

from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size=0.2)

Y_train = train.iloc[:,0]

X_train = train.iloc[:,1:]
X_train
X_train = X_train.to_numpy()

X_test = test.to_numpy()

Y_train = Y_train.to_numpy()
temp = X_train.reshape(-1,20,20)

X_test = X_test.reshape(-1,20,20)

print(temp.shape)

print(X_test.shape)
w=14

h=14

fig=plt.figure(figsize=(w,h))

columns = 4

rows = 5

for i in range(1, rows*columns+1):

    img1 = temp[i+random.randrange(1,300)]

    fig.add_subplot(rows, columns, i)

    plt.imshow(img1)

plt.show()
w=14

h=14

fig=plt.figure(figsize=(w,h))

columns = 2

rows = 5

for i in range(1, rows+1,2):

    img1 = temp[i+random.randrange(1,300)]

    img = Image.fromarray(img1.astype('uint8'))

    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150))

    fig.add_subplot(rows, columns, i)

    plt.imshow(img1)

    plt.title('Before'), plt.xticks([]), plt.yticks([])

    fig.add_subplot(rows, columns, i+1)

    plt.imshow(img)

    plt.title('After'), plt.xticks([]), plt.yticks([])

plt.show()
def sharpner(img):

    img = Image.fromarray(img.astype('uint8'))

    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150))

    return np.array(img)



for i in range(temp.shape[0]):

    temp[i] = sharpner(temp[i])

for i in range(X_test.shape[0]):

    X_test[i] = sharpner(X_test[i])

    

print(temp.shape)

print(X_test.shape)





plt.imshow(temp[3])
X_train = temp.reshape(-1,20,20,1)

X_test = X_test.reshape(-1,20,20,1)
print(X_train.shape)

print(X_test.shape)

print(Y_train.shape)
X_train = X_train/255.

X_test = X_test/255.

print(X_train.shape)

print(X_test.shape)
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



Y_train = one_hottie(Y_train, 10)

print ("Y shape: " + str(Y_train.shape))
model = tf.keras.Sequential([

    tf.keras.layers.Conv2D(64, 3, activation='relu', input_shape=(20,20,1),padding="same"),

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
model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=0.006),

              loss=tf.keras.losses.CategoricalCrossentropy(),

              metrics=['accuracy'])



result = model.fit(x=X_train,y=Y_train,batch_size=130,epochs=30,verbose=1,shuffle=False,initial_epoch=0,

                   validation_split=0.2)
#plt.plot(result.history['acc'], label='train')

#plt.plot(result.history['val_acc'], label='valid')

#plt.legend(loc='upper left')

#plt.title('Model accuracy')

#plt.ylabel('Accuracy')

#plt.xlabel('Epoch')

#plt.show()

plt.plot(result.history['loss'], label='train')

plt.plot(result.history['val_loss'], label='test')

plt.legend(loc='upper right')

plt.title('Model Cost')

plt.ylabel('Cost')

plt.xlabel('Epoch')

plt.show()
model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=0.0001),

              loss=tf.keras.losses.CategoricalCrossentropy(),

              metrics=['accuracy'])



result = model.fit(x=X_train,y=Y_train,batch_size=128,epochs=50,verbose=1,shuffle=False,initial_epoch=20,

                   validation_split=0.2)
#plt.plot(result.history['acc'], label='train')

#plt.plot(result.history['val_acc'], label='valid')

#plt.legend(loc='upper left')

#plt.title('Model accuracy')

#plt.ylabel('Accuracy')

#plt.xlabel('Epoch')

#plt.show()

plt.plot(result.history['loss'], label='train')

plt.plot(result.history['val_loss'], label='test')

plt.legend(loc='upper right')

plt.title('Model Cost')

plt.ylabel('Cost')

plt.xlabel('Epoch')

plt.show()
model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=0.00006),

              loss=tf.keras.losses.CategoricalCrossentropy(),

              metrics=['accuracy'])



result = model.fit(x=X_train,y=Y_train,batch_size=64,epochs=100,verbose=1,shuffle=False,initial_epoch=50,

                   validation_split=0.2)
check = model.evaluate(X_train )
preds = model.predict_classes(X_train)

preds.shape
train = pd.read_csv("../input/dataset-for-test/dataset_00_with_header.csv")

Y_train = train.iloc[:,0]

Y_train = Y_train.to_numpy()
#conf = tf.math.confusion_matrix(preds,Y_train)
with tf.Session() as session:

    print(conf.eval())
preds = model.predict_classes(X_test)
preds.shape
arr = [x for x in range(1,28001)]

label = pd.DataFrame(arr,columns = ["ImageId"])

label["Label"] = pd.DataFrame(preds)

label.head()
label.to_csv('Y_test.csv',header=True,index = False)
model.save("saved_model")