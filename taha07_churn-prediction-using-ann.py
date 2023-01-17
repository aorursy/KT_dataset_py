# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/churn-modelling/Churn_Modelling.csv")

df.head()
df.info()
df.isnull().sum()
plt.rcParams['figure.figsize'] =(10,8)

plt.style.use('classic')

color = plt.cm.PuRd(np.linspace(0,1,3))

df['Geography'].value_counts().plot.bar(color=color);
plt.rcParams['figure.figsize'] =(10,8)

plt.style.use('classic')

color = plt.cm.PuRd(np.linspace(0,1,3))

sns.countplot(x='Geography',hue='Exited',data=df);
plt.rcParams['figure.figsize'] =(10,8)

plt.style.use('classic')

color = plt.cm.PuRd(np.linspace(0,1,3))

sns.countplot(x='Gender',hue='Exited',data=df);
plt.rcParams['figure.figsize'] =(10,8)

plt.style.use('classic')

color = plt.cm.PuRd(np.linspace(0,1,3))

sns.countplot(x='Gender',hue='HasCrCard',data=df)
salary = df[['EstimatedSalary','Exited']].nlargest(100,'EstimatedSalary')

salary['Exited'].value_counts()
salary_small = df[['EstimatedSalary','Exited']].nsmallest(100,'EstimatedSalary')

salary_small['Exited'].value_counts()
plt.rcParams['figure.figsize'] =(10,8)

plt.style.use('classic')

color = plt.cm.PuRd(np.linspace(0,1,3))

sns.countplot(x='Gender',hue='NumOfProducts',data=df)
facet = sns.FacetGrid(df,hue="Exited",aspect = 4)

facet.map(sns.kdeplot,"Age",shade = True)

facet.set(xlim = (0,df["Age"].max()))

facet.add_legend()

plt.show()
facet = sns.FacetGrid(df,hue="Exited",aspect = 4)

facet.map(sns.kdeplot,"Tenure",shade = True)

facet.set(xlim = (0,df["Tenure"].max()))

facet.add_legend()

plt.show()
df.drop(['RowNumber','CustomerId','Surname'],axis=1,inplace=True)

df = pd.get_dummies(df,drop_first=True)

df.head(2)
x = df.drop('Exited',axis=1)

y = df['Exited']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)

x_test = scaler.transform(x_test)
import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import Sequential

from tensorflow.keras.layers import Flatten,Dense
model = Sequential()

model.add(Dense(x.shape[1],activation='relu',input_dim=x.shape[1]))

model.add(Dense(128,activation='relu'))

model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
history =  model.fit(x_train,y_train.to_numpy(),batch_size=8,epochs=10,verbose=1,validation_split=0.2)
model.evaluate(x_test,y_test)
y_pred =model.predict_classes(x_test)
from sklearn.metrics import accuracy_score,confusion_matrix

accuracy_score(y_test,y_pred)
#accuracy

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model Accuracy')

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend(['Train','Val'],loc='upper left')

plt.show()



#Loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model Loss')

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend(['Train','Val'],loc='upper left')

plt.show()
from mlxtend.plotting import plot_confusion_matrix

import matplotlib
class_names=['Staying','Leaving']

matrix = confusion_matrix(y_test,y_pred)

plot_confusion_matrix(conf_mat= matrix,figsize=(5,5),class_names=class_names,show_normed=True)

plt.xticks(rotation=0);