# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

train.shape

test.shape
train_len=len(train)
print(train_len)
test_pass_ID=test['PassengerId']
df = pd.concat(objs=[train,test],axis=0,sort=False).reset_index(drop=True)
df.head()
df.isnull().sum()
df1= df.fillna(np.nan)
df1.isna().sum()
df1['Age']=df1['Age'].fillna(df1['Age'].median())
df1.isna().sum()
df1.groupby('Cabin').count()
df1.describe()
df1.info()
df1['Cabin_grp']= df1['Cabin'].astype(str).str[0]
df1['Cabin_grp'].nunique()
df2 = df1.drop('Cabin',axis=1)
df2.head(20
    )
df2['Embarked'].mode()
df2['Embarked'] = df2['Embarked'].fillna("S")
df2.isna().sum()
df2['Fare'].median()
df2['Fare']= df2['Fare'].fillna(df2['Fare'].median())
df2.head()
df3= df2
df3.head()
df3= df3.drop(["Name","Ticket"],axis=1)
df3= pd.get_dummies(df3)
df3.head()

df3_train = df3[:train_len]
df3_test= df3[train_len:]
df3_train.shape
df3_train.head()
df3_test.head(25)
df3_test.shape
df3_test= df3_test.drop('Survived',axis=1)
df3_test.shape
df3_train_Y= df3_train['Survived'].astype(int)
df3_train_X= df3_train.drop('Survived', axis=1)
df3_train_X.shape
df3_train_Y.shape
df3_train_Y.head()
knn = KNeighborsClassifier()
knn.fit(df3_train_X, df3_train_Y)
Y_pred= knn.predict(df3_test)
Y_pred

Y_pred_Ser= pd.Series(Y_pred,name= "Survived")
my_res= pd.concat([test_pass_ID,Y_pred_Ser],axis=1)
my_res.head(15)
my_res.to_csv("Titanic_Pred_Sub.csv",index=False)
logreg = LogisticRegression()
logreg.fit(df3_train_X,df3_train_Y)
Y_pred_logreg= logreg.predict(df3_test)
Y_pred_logreg_Ser= pd.Series(Y_pred_logreg, name= "Survived")
my_res_logreg = pd.concat([test_pass_ID, Y_pred_logreg_Ser], axis=1)
my_res_logreg.to_csv("Titanic_logreg_Sub.csv", index= False)
dectree= DecisionTreeClassifier()
dectree.fit(df3_train_X,df3_train_Y)
pred_dectree = dectree.predict(df3_test)
pred_dectree_Ser= pd.Series(pred_dectree, name= "Survived")
res_dectree= pd.concat([test_pass_ID, pred_dectree_Ser], axis=1)
res_dectree.to_csv("res_dectree.csv", index= False)
