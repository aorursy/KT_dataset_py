# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# import the libraries:
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import dataset
df=pd.read_csv("../input/Churn_Modelling.csv")
df.head()
#df.describe()
#df.info()
# dataset visualization:
sns.pairplot(df,hue='Gender')
sns.scatterplot(df['Geography'],df['Age'],hue='Gender',data=df)
sns.countplot(df['Geography'],hue='Gender',data=df)
sns.distplot(df['Age'],bins=20,kde=False)
# spliting the dataset into the dependent and independet form:
X=df.iloc[:,3:13].values
y=df.iloc[:,13].values
X
y
# ecoding the dataset:
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
lableencoder_X_1=LabelEncoder()
X[:,1]=lableencoder.fit_transform(X[:,1])
lableencoder_X_2=LabelEncoder()
X[:,2]=lableencoder_X_2.fit_transform(X[:,2])
onehotencoder=OneHotEncoder(categorical_features=[1])
X=onehotencoder.fit_transform(X).toarray()
X=X[:,1:]
X
# sliting the datset into form train and test from :
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
#feature scaling:
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)
# import the keras:
import keras
from keras.models import Sequential
from keras.layers import Dense

# initialisation the ANN
classifier=Sequential()
# adding the input layer and first hidden layer:
classifier.add(Dense(output_dim=6,activation='relu',init='uniform',input_dim=11))
# adding the second hidden layer :
classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
# adding the output layare :
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
# compiling the ANN:
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
# fitting the model on training set
classifier.fit(X_train,y_train,batch_size=100,epochs=1000)
# making the prediction of new result:
y_pred=classifier.predict(X_test)
y_pred=(y_pred>0.5)

y_pred
# making the confusion matrix:
from sklearn.metrics import confusion_matrix,classification_report
cm=confusion_matrix(y_test,y_pred)
cm
cr=classification_report(y_test,y_pred)
cr
(1519+205)/2000
