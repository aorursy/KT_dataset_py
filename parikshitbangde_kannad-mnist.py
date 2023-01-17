# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt 

import seaborn as sns 

import tensorflow as tf 

import keras 
print ("we are usint tensorflow version :" ,tf.__version__)
train = pd.read_csv("../input/Kannada-MNIST/train.csv")
test = pd.read_csv("../input/Kannada-MNIST/test.csv")

Dig = pd.read_csv("../input/Kannada-MNIST/Dig-MNIST.csv")
train.head()
Dig.head()
print ("training shape ",train.shape,'\n',"testing shape ",test.shape,'\n',"Dig shape ",Dig.shape)
X = train.drop(columns=['label'],axis=1)

X_val = Dig.drop(columns=['label'],axis=1)
y = train['label']

y_val = Dig['label']
print (X.shape,X_val.shape)
print (y.shape,y_val.shape)
X = X / 255.0

X_val = X_val / 255.0
y = tf.keras.utils.to_categorical(y, num_classes=10)

y_val = tf.keras.utils.to_categorical(y_val, num_classes=10)
# visualizing the first 10 images in the dataset and their labels

%matplotlib inline

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 1))

for i in range(10):

    plt.subplot(1, 10, i+1)

    plt.imshow(X.iloc[i].values.reshape(28, 28), cmap="gray")

    plt.axis('off')

    print('label for each of the below image: %s' % (np.argmax(y[0:10][i])))

plt.show()
sns.countplot(train['label'])
X =np.array(X).astype('float32')

y = np.array(y).astype('float32')

X_val =np.array(X_val).astype('float32')

y_val = np.array(y_val).astype('float32')
print ('traing set',X.shape,y.shape)

print ('test set',X_val.shape,y_val.shape)
model = tf.keras.models.Sequential() # Instantiating keras sequential models from keras 

Lambda = 0.001

# First layer (input layer) of  28*28 = 784 after flattening the image of 28 * 28 picxels

model.add(tf.keras.layers.Dense(784,input_dim=784,kernel_initializer='uniform', activation='relu'))

# second layer 

model.add(tf.keras.layers.Dense(181, kernel_initializer='uniform', activation='relu'))

# third layer

#model.add(tf.keras.layers.Dense(181, kernel_initializer='uniform', activation='relu',kernel_regularizer=tf.keras.regularizers.l2(Lambda)))

#fourth layer 

#model.add(tf.keras.layers.Dense(90, kernel_initializer='uniform', activation='relu',kernel_regularizer=tf.keras.regularizers.l2(Lambda)))

# Final layer with activation function as softmax and 10 neurons 

model.add(tf.keras.layers.Dense(10, activation='softmax'))







# Create optimizer with non-default learning rate

sgd_optimizer =  tf.keras.optimizers.SGD(lr=0.02, decay=1e-6, momentum=0.9)



# Compile the model

model.compile(optimizer=sgd_optimizer, loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X,y,validation_data=(X_val,y_val),epochs=50)
id = test['id']

test = test.drop(columns=['id'])

predict = model.predict(test)

labels = np.argmax(predict,axis=1)
my_submission = pd.DataFrame({'id': id, 'label': labels})
my_submission.to_csv('submission.csv', index=False)
my_submission
sns.distplot(my_submission['label'])
my_submission[my_submission['label']==3]
my_submission.shape