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
import tensorflow as tf

import pandas as pd

import numpy as np

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from matplotlib import pyplot as plt
train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv") 

test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
#plot frequencies for the lables in the train dataframe



ax = sns.countplot(y="label",data=train)

for i in ax.patches:

    ax.text(i.get_width(), i.get_y()+i.get_height()/2,i.get_width(), ha='left')
#Helper function to read the training data



def extract_label_pixel(i,img_height, img_length):

    pixel = np.array(train[train.columns[1:]].iloc[i]).reshape(img_height, img_length)

    label = np.array(train[train.columns[0]].iloc[i])

    return pixel, label
#print 10 random sample for each digit



fig1, ax1 = plt.subplots(10,10, figsize=(15,20))

for v in range(10):

    sample =  train[(train.label == v)].sample(n=10, replace=False).index

    for i,j in enumerate(sample):

        x, y = extract_label_pixel(j,28,28)

        ax1[v][i].imshow(x, cmap="gray_r")

        ax1[v][i].axis('off')

#         ax1[v][i].set_title(y)    
#Normalizing data to feed into tensorflow



y_train = train["label"]

x_train = train.drop(labels = ["label"],axis = 1)

x_train = x_train / 255.0

x_train = x_train.values.reshape(-1,28,28,1)



x_test = test / 255.0

x_test = x_test.values.reshape(-1,28,28,1)



x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.10, random_state=35)
#setting up tensorflow model



model = tf.keras.models.Sequential([

        tf.keras.layers.Conv2D(196, (3,3), activation='relu', input_shape=(28, 28, 1)),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(98, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(98, (3,3), activation='relu'),

    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(49, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(10, activation='softmax')

])



model.summary()



model.compile(loss = 'sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=20, steps_per_epoch=25, validation_data = (x_val, y_val), verbose = 1, shuffle = True)
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(acc))



plt.plot(epochs, acc, 'r', label='Training accuracy')

plt.plot(epochs, val_acc, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend(loc=0)

plt.figure()





plt.show()
y_pred = np.argmax(model.predict(x_val),axis = 1) 



confusion_mtx = confusion_matrix(np.array(y_val), y_pred) 

# # plot the confusion matrix

f,ax = plt.subplots(figsize=(8, 8))

sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Blues",linecolor="gray", fmt= '.1f',ax=ax)

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.title("Confusion Matrix")

plt.show()
pred = model.predict(x_test)

pred = np.argmax(pred,axis = 1)

pred = pd.Series(pred,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),pred],axis = 1)

submission.to_csv("submissions.csv",index=False)