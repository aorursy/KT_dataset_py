# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns

import warnings as wrn



wrn.filterwarnings('ignore')

sns.set_style("whitegrid")



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')

data.head()
data.info()
data["Pregnancies"].value_counts()
fig,ax = plt.subplots(figsize=(10,7))

sns.countplot(data["Pregnancies"])

plt.show()
data["Glucose"].head()
fig,ax = plt.subplots(figsize = (10,7))

sns.distplot(data["Glucose"],color="#FE5205")

plt.show()
data["BloodPressure"].head(10)
fig,ax = plt.subplots(figsize=(10,7))

sns.distplot(data["BloodPressure"],color="#00B037")

plt.show()
data["SkinThickness"].head(10)
fig,ax = plt.subplots(figsize=(10,7))

sns.distplot(data["SkinThickness"],color="#C0F714")

plt.show()
data["Insulin"].head(10)
fig,ax = plt.subplots(figsize=(10,7))

sns.distplot(data["Insulin"],color="#077F8F")

plt.show()
data["BMI"].head(10)
plt.subplots(figsize=(10,7))

sns.distplot(data["BMI"],color="#DB6A14")

plt.show()
data["DiabetesPedigreeFunction"].head()
fig,ax = plt.subplots(figsize=(10,7))

sns.distplot(data["DiabetesPedigreeFunction"],color="#8F105A")

plt.show()
data["Age"].head(10)
fig,ax = plt.subplots(figsize=(10,7))

sns.distplot(data["Age"],color="#DB620D")

plt.show()
data["Outcome"].value_counts()
fig,ax = plt.subplots(figsize=(10,7))

sns.countplot(data["Outcome"])

plt.show()
def outlier_dropper(dataset):

    check_index = []

    final_index = []

    for feature in dataset: # Each iteration is a different feature

        

        Q1 = dataset[feature].describe()["25%"] # Lower Quartile

        Q3 = dataset[feature].describe()["75%"] # Upper Quartile

        

        IQR = Q3-Q1

        STEP = IQR*1.5

        

        

        indexes = data[(data[feature]<Q1-IQR) | (data[feature]>Q3+IQR)].index.values # Taking outlier's indexes.

        

        for i in indexes:  

            check_index.append(i) # Appending each index into the check_index list.

    

    for i in check_index:        

        check_index.remove(i)

        if i in check_index: # If i still exists (If there is two outliers in the i index)

            final_index.append(i) # Append it.

    

    return np.unique(final_index)
indexes = outlier_dropper(data)

print(indexes)

print("------------------------------------------------------------------------------")

print(len(indexes))
data.drop(indexes,inplace=True)
data.info()
fig,ax = plt.subplots(figsize=(8,8))

sns.heatmap(data.corr(),annot=True,fmt=".2f",linewidths=1.5)

plt.show()
fig = plt.figure(figsize=(7,5))

fig.add_subplot(1,2,1)

sns.kdeplot(data["Glucose"],data["Outcome"])

fig.add_subplot(1,2,2)

sns.scatterplot(data["Glucose"],data["Outcome"])

plt.show()
fig = plt.figure(figsize=(7,5))

fig.add_subplot(1,2,1)

sns.kdeplot(data["Outcome"],data["Age"])

fig.add_subplot(1,2,2)

sns.scatterplot(data["Outcome"],data["Age"])

plt.show()
fig = plt.figure(figsize=(7,5))

fig.add_subplot(1,2,1)

sns.kdeplot(data["BMI"],data["Outcome"])

fig.add_subplot(1,2,2)

sns.scatterplot(data["BMI"],data["Outcome"])

plt.show()
fig,ax = plt.subplots(figsize=(10,7))

sns.countplot(data["Pregnancies"])

plt.show()
pregnancies = []



for i in data["Pregnancies"]:

    

    if i==11 or i==12 or i==13 or i==14 or i==15 or i==17:

        pregnancies.append(11)

    

    else:

        pregnancies.append(i)



data.Pregnancies = pregnancies
fig,ax = plt.subplots(figsize=(10,7))

sns.countplot(data["Pregnancies"])

plt.show()
data = pd.get_dummies(data,columns=["Pregnancies"])

data.head()
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler(feature_range=(0,1))



x = data.drop("Outcome",axis=1)

y = data.Outcome



x = scaler.fit_transform(x)
print("Shape of x",x.shape)

y = y.values

print("Shape of y",y.shape)
y = y.reshape(-1,1)
from sklearn.model_selection import train_test_split



x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=1)
from keras.layers import Dropout,Dense

from keras.models import Sequential
model = Sequential()

model.add(Dense(units=16,kernel_initializer="uniform",activation="tanh",input_dim=19)) # Layer 1

model.add(Dropout(0.25))



model.add(Dense(units=16,kernel_initializer="uniform",activation="tanh")) # Layer 2

model.add(Dropout(0.50))



model.add(Dense(units=32,kernel_initializer="uniform",activation="tanh")) # Layer 3

model.add(Dropout(0.50))



model.add(Dense(units=32,kernel_initializer="uniform",activation="tanh")) # Layer 4 

model.add(Dropout(0.50))



model.add(Dense(units=32,kernel_initializer="uniform",activation="tanh")) # Layer 5

model.add(Dropout(0.50))



model.add(Dense(units=32,kernel_initializer="uniform",activation="tanh")) # Layer 6

model.add(Dropout(0.50))



model.add(Dense(units=32,kernel_initializer="uniform",activation="tanh")) # Layer 7

model.add(Dropout(0.50))



model.add(Dense(units=1,kernel_initializer="uniform",activation="sigmoid")) # Output Layer

model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
model.fit(x_train,y_train,epochs=250)
from sklearn.metrics import accuracy_score

y_head = model.predict_classes(x_test)



print("The score is ",accuracy_score(y_test,y_head))
from sklearn.metrics import confusion_matrix



confusion_matrix = confusion_matrix(y_test,y_head)



fig,ax = plt.subplots(figsize=(6,6))

sns.heatmap(confusion_matrix,annot=True,fmt="0.1f",cmap="Greens_r",linewidths=1.5)

plt.show()