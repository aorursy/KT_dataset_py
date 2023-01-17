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
train = pd.read_csv("../input/train.csv")
train.head()
train.shape
train.describe
test=pd.read_csv("../input/test.csv")
test.head()
test.shape
import matplotlib.pyplot as plt
%matplotlib inline
sex_pivot = train.pivot_table(index="Sex",values="Survived")
sex_pivot.plot.bar()
plt.show()
pclass_pivot = train.pivot_table(index="Pclass",values='Survived')
pclass_pivot.plot.bar()
plt.show()
train["Age"].describe()
survived = train[train["Survived"]==1]
died = train[train["Survived"]==0]
survived["Age"].plot.hist(alpha=0.5,color = 'red',bins =50)
died["Age"].plot.hist(alpha=0.5,color='blue',bins =50)
plt.legend(['survived','died'])
plt.show()
def process_age(df,cut_points,label_names):
    df["Age"] = df["Age"].fillna(-0.5)
    df["Age_cat"] = pd.cut(df["Age"],cut_points,labels=label_names)
    return df
cut_points = [-1,0,5,12,18,35,60,100]
label_names = ["Missing","Infant","Child","Teen","Young","Adult","Senior"]

train=process_age(train,cut_points,label_names)
test =process_age(test,cut_points,label_names)

pivot = train.pivot_table(index = "Age_cat",values ="Survived")
pivot.plot.bar()
plt.show()
train["Pclass"].value_counts()
def create_dummy(df,column_name):
    dummy = pd.get_dummies(df[column_name],prefix = column_name)
    df = pd.concat([df,dummy],axis =1)
    return df

for col in ["Pclass","Sex","Age_cat"]:
    train = create_dummy(train,col)
    test = create_dummy(test,col)
train.head()
from sklearn.linear_model import LogisticRegression
columns = ['Pclass_1','Pclass_2','Pclass_3','Sex_female','Sex_male','Age_cat_Missing','Age_cat_Infant','Age_cat_Child','Age_cat_Teen','Age_cat_Young','Age_cat_Adult','Age_cat_Senior']
lr = LogisticRegression()
lr.fit(train[columns],train["Survived"])
hold =test
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(train[columns],train["Survived"],test_size=0.20,random_state =0)
pred = lr.fit(X_train,Y_train).predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test,pred))
holdout_pred = lr.predict(hold[columns])
holdout_id = hold["PassengerId"]
sub_df = {"PassengerId":holdout_id,"Survived":holdout_pred}
submission = pd.DataFrame(sub_df)
submission.to_csv("submission_csv",index=False)
