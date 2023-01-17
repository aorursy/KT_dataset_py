# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/voicegender/voice.csv")
df.info()
sns.countplot(df.label,palette="coolwarm")

plt.show()

df.label.value_counts()
df.label = [1 if i == "male" else 0 for i in df.label]
df.label
X=df.drop(["label"],axis=1)

y=df.label.values
x=(X-np.min(X))/(np.max(X)-np.min(X))
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.2,random_state=42)

print("X train shape : ",X_train.shape)

print("Y train shape : ",Y_train.shape)

print("X test shape : ",X_test.shape)

print("Y test shape : ",Y_test.shape)
from keras.models import Sequential

from keras.layers import Dense , Dropout
classifier= Sequential() # start the model

classifier.add(Dense(output_dim=80,init="uniform",activation="relu",input_dim=20))

classifier.add(Dropout(p=0.2))

classifier.add(Dense(output_dim=100,init="uniform",activation="tanh"))

classifier.add(Dropout(p=0.2))

classifier.add(Dense(output_dim=120,init="uniform",activation="relu"))

classifier.add(Dropout(p=0.2))

classifier.add(Dense(output_dim=200,init="uniform",activation="relu"))

classifier.add(Dropout(p=0.2))

classifier.add(Dense(output_dim=1,init="uniform",activation="sigmoid"))
classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
classifier.fit(X_train.values,Y_train,batch_size=250,nb_epoch=1000)
y_prediction=classifier.predict(X_test)
y_prediction= [1 if i>=0.5 else 0 for i in y_prediction]
from sklearn.metrics import confusion_matrix

cfm = confusion_matrix(Y_test,y_prediction)
f,ax=plt.subplots(figsize=(10,10))

sns.heatmap(cfm,annot=True,cmap="coolwarm",linewidths=2,linecolor="black",fmt=".1f",ax=ax)

plt.show()