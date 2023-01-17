import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))

Tit1=pd.read_csv("../input/Titanic.csv")

# Any results you write to the current directory are saved as output.
#Check for null values

Tit1.isnull().sum()
Tit2= Tit1.copy()

Tit2.head()
#Drop all the null values to prepare the training dataset

Tit3=Tit2.dropna()

Tit3.head()
X_train=Tit3[['Survived','Pclass','SibSp','Parch','Fare']]

X_train.head()
Y_train=Tit3[['Age']]

Y_train.head()
#Apply linear regression 

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, Y_train)

Tit2.shape, Tit3.shape
#Pick up the data having invalid age

Tit4=Tit2.copy()

cond1=Tit4['Age']>0

Tit5=Tit4[cond1]

Tit6=pd.concat([Tit5,Tit2]).drop_duplicates(keep=False)

Tit6.head()

X_test=Tit6[['Survived','Pclass','SibSp','Parch','Fare']]

X_test.head()
Y_test=Tit6[['Age']]

Y_test.head()
#Predict the age

y_pred=np.round(model.predict(X_test),2)

Tit6['Age']=y_pred
Tit6.head()
Tit6.shape, Tit4.shape
#Recreate the final dataset



Tit7=pd.concat([Tit6,Tit5]).drop_duplicates(keep=False)

Tit7.head()
Tit7.shape
Tit7.isnull().sum()