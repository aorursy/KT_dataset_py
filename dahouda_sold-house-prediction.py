# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# Visualization libraries

import seaborn as sns

import matplotlib.pyplot as plt



import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers



# Spliting data and creating model libraries

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from keras.models import Sequential #initialize neural network library

from keras.layers import Dense #build our layers library

from tensorflow import keras

from keras.models import Sequential



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data_train = pd.read_csv("../input/house-price2/House-Price2.csv")

data_train.head()
data_train.info()
# We can observe number of sold House

sns.countplot(data_train["Sold"])

plt.show()
data_train.describe()
# Let's plot the distribution of Hot Room and sold

sns.jointplot(x='n_hot_rooms', y='Sold', data=data_train)
sns.countplot(x='airport', data=data_train)
sns.countplot(x='waterbody', data=data_train)
sns.countplot(x='bus_ter', data=data_train)
data_train.info()
np.percentile(data_train.n_hot_rooms,[99])
np.percentile(data_train.n_hot_rooms,[99])[0]
nv = np.percentile(data_train.n_hot_rooms,[99])[0]
data_train[(data_train.n_hot_rooms > nv)]
data_train.n_hot_rooms[(data_train.n_hot_rooms > 3 * nv)] = 3 * nv
data_train[(data_train.n_hot_rooms > nv)]
np.percentile(data_train.rainfall,[1])[0]
lv = np.percentile(data_train.rainfall,[1])[0]
data_train[(data_train.rainfall < lv)]
data_train[(data_train.rainfall < lv)]
data_train.info()
#Impute Missing values for 1 columns

data_train.n_hos_beds = data_train.n_hos_beds.fillna(data_train.n_hos_beds.mean())

# For all columns : df = df.fillna(df.mean())
data_train.info()
data_train.head()
data_train['avg_dist'] = (data_train.dist1 + data_train.dist2 + data_train.dist3 + data_train.dist4) / 4
data_train.describe()
del data_train['dist1']
del data_train['dist2']
del data_train['dist3']
del data_train['dist4']
data_train.head()
data_train.shape
del data_train['bus_ter']
data_train = pd.get_dummies(data_train)
data_train.head()
data_train.shape
del data_train['airport_NO']
del data_train['waterbody_None']
data_train.head()
data_train.shape
data_train.corr()
data_test = data_train
data_test.shape
X = data_train.drop(["Sold"],axis=1)

Y = data_train["Sold"]

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2)

print("x_train shape: ",x_train.shape)

print("y_train shape: ",y_train.shape)

print("x_test shape: ",x_test.shape)

print("y_test shape: ",y_test.shape)
X.shape
my_model = Sequential() # initialize neural network

my_model.add(Dense(units = 128, activation = 'relu', input_dim = X.shape[1]))

my_model.add(Dense(units = 32, activation = 'relu'))

my_model.add(Dense(units = 16, activation = 'relu'))

my_model.add(Dense(units = 8, activation = 'relu'))

my_model.add(Dense(units = 4, activation = 'relu'))

my_model.add(Dense(units = 1, activation = 'sigmoid')) #output layer

my_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
my_model.summary()
keras.utils.plot_model(my_model)
model = my_model.fit(x_train,y_train,epochs=750)

mean = np.mean(model.history['accuracy'])

print("Accuracy mean: "+ str(mean))
model.params
pd.DataFrame(model.history).plot(figsize=(8,5))

plt.grid(True)

plt.gca().set_ylim(0,1)

plt.show()
y_predict = my_model.predict(X)

cm = confusion_matrix(Y,np.argmax(y_predict, axis=1))



f, ax = plt.subplots(figsize=(5, 5))

sns.heatmap(cm, annot=True, fmt="d", linewidths=.5, ax=ax)
ids = data_test['House_id']

#predict = classifier.predict(data_test_x)

predict = my_model.predict(X)



#set the output as a dataframe and convert to csv file named submission.csv

output = pd.DataFrame({ 'House_id' : ids, 'Sold': np.argmax(predict,axis=1)})

output.to_csv('submission.csv', index=False)
submission = pd.read_csv('submission.csv', index_col=0)

submission.head(50)