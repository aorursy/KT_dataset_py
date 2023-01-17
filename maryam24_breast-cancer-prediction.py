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

import os

import tensorflow as tf

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from IPython.display import clear_output

from time import sleep

from sklearn.model_selection import train_test_split



import functools, datetime


data = pd.read_csv("../input/breast-cancer-wisconsin-data/data.csv")
data.info()
data.head(10)
data = data.drop(["id", "Unnamed: 32"], axis = 1)

data.head(10)
sns.countplot(data["diagnosis"])
plt.figure(figsize  = (25,20))

plt.title("Correlation Matrix")

sns.heatmap(data.corr(), annot = True)
fig = plt.figure(figsize = (20,15))

plt.subplot(231)

sns.scatterplot(x = data['radius_mean'], y = data['texture_mean'], hue = "diagnosis", data = data,  palette = "husl")

plt.title("radius mean vs texture mean")

plt.subplot(232)

sns.scatterplot(x = data['radius_mean'], y = data['smoothness_mean'], hue = 'diagnosis', data= data)

plt.title('radius mean vs smoothness mean')

plt.subplot(233)

sns.scatterplot(x= data['radius_mean'], y = data['symmetry_mean'], hue = 'diagnosis', data = data)

plt.title('radius mean vs symmetry mean')

fig.suptitle("Correlation < 0.5")
fig = plt.figure(figsize = (20,15))

plt.subplot(231)

sns.scatterplot(x = data['radius_mean'], y = data['compactness_mean'], hue = "diagnosis", data = data,  palette = "husl")

plt.title("radius mean vs compactness mean")

plt.subplot(232)

sns.scatterplot(x = data['radius_mean'], y = data['concavity_mean'], hue = 'diagnosis', data= data)

plt.title('radius mean vs concavity mean')

plt.subplot(233)

sns.scatterplot(x= data['radius_mean'], y = data['concave points_mean'], hue = 'diagnosis', data = data)

plt.title('radius mean vs concave points mean')

fig.suptitle("0.5 < Correlation < 0.9")
def preprocessing(data, feature):

    featureMap=dict()

    count=0

    for i in sorted(data[feature].unique(),reverse=True):

        featureMap[i]=count

        count=count+1

    data[feature]=data[feature].map(featureMap)

    x_raw=data.drop([feature],axis=1)

    y_raw=data[feature]

    return x_raw, y_raw
X, y = preprocessing(data, feature = 'diagnosis')
y = np.array(y)


x_train, xtest, y_train, ytest = train_test_split(X, y, test_size = 0.33, random_state = 42)

# print(y_train.shape)



num_inputs = x_train.shape[1]
y_train.shape
input_dim = tf.keras.layers.Input

fc = functools.partial(tf.keras.layers.Dense)
def model():

    inputs = input_dim(shape = num_inputs)

    fc1 = fc(32)(inputs)

    fc2 = fc(64)(fc1)

    fc3 = fc(1)(fc2)

    

    fullmodel = tf.keras.Model(inputs, fc3)

    

    optimizers = tf.keras.optimizers.Adam(learning_rate=0.1)

    

    fullmodel.compile(optimizer=optimizers, loss='mean_squared_error', metrics=['mae', 'acc'])

    

    return fullmodel

    

    
model1 = model()

model1.summary()
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(filepath='./weights.hdf5', verbose=1, save_best_only=True),

    tensorboard_callback,

    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

]
history = model1.fit(x_train, y_train, epochs = 100)

plt.plot(history.history['loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()
scores = model1.evaluate(xtest, ytest)
print("Loss:",scores[0])

print("Accuracy",scores[1]*100)