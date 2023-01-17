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
test = pd.read_csv("../input/test.csv")
test.head()
train = pd.read_csv("../input/train.csv")
train.head()
print("Shape of train: {}".format(train.shape))
print("Shape of test: {}".format(test.shape))
total = pd.DataFrame(train.isnull().sum().sort_values(ascending=False), columns=['Total'])
percentage = pd.DataFrame(round(100*(train.isnull().sum()/train.shape[0]),2).sort_values(ascending=False)\
                          ,columns=['Percentage'])
pd.concat([total, percentage], axis = 1)
import matplotlib.pyplot as plt
%matplotlib inline
sex_pivot=train.pivot_table(values='Survived', index='Sex', aggfunc='mean',margins='True', margins_name='Total')
print(sex_pivot)
sex_pivot.plot.bar()
plt.show()
class_pivot=train.pivot_table(values='Survived', index='Pclass', aggfunc='mean',margins='True', margins_name='Total')
print(class_pivot)
class_pivot.plot.bar()
plt.show()
train.Age.describe()
survived = pd.DataFrame(train.loc[train["Survived"] == 1].groupby('Age')['Survived'].count())
dead = pd.DataFrame(train.loc[train["Survived"] == 0].groupby('Age')['Survived'].count())
col_rename = {'Survived':'Dead'}
dead = dead.rename(columns=col_rename)
total=pd.concat([survived,dead], axis=1)
total.plot.bar()
plt.show()
survived = train.loc[train["Survived"] == 1]
dead = train.loc[train["Survived"] == 0]
survived['Age'].plot.hist(alpha=0.5,color='blue',bins=50)
dead['Age'].plot.hist(alpha=0.5,color='green',bins=50)   
plt.legend(['Survived','Died'])
plt.show()
def process_age(df,cut_points,label_names):
    df['Age']=df['Age'].fillna(-0.5)
    age_index = df.columns.get_loc('Age') + 1
    df.insert(loc=age_index,column='Age_Categories',value=pd.cut(df['Age'],cut_points,labels=label_names))
    return df

cut_points = [-1,0,5,12,19,35,60,100]
label_names = ["Missing","Infant","Child","Teenager","Young Adult","Adult","Senior"]

train = process_age(train,cut_points,label_names)
test = process_age(test,cut_points,label_names)
pivot = train.pivot_table(values='Survived',index='Age_Categories',aggfunc='mean')
print(pivot)
pivot.plot.bar()
plt.show()
def create_dummies(df, column_name):
    dummies = pd.get_dummies(df[column_name],prefix=column_name)
    df=pd.concat([df,dummies],axis=1)
    return df

for column in ['Age_Categories','Sex','Pclass']:
    train = create_dummies(train, column)
    test = create_dummies(test, column)

train.head()
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
columns = ['Pclass_1','Pclass_2','Pclass_3','Sex_female','Sex_male','Age_Categories_Missing','Age_Categories_Infant','Age_Categories_Child','Age_Categories_Teenager','Age_Categories_Young Adult','Age_Categories_Adult','Age_Categories_Senior']
lr.fit(train[columns], train["Survived"])
holdout = test # from now on we will refer to this
               # dataframe as the holdout data

from sklearn.model_selection import train_test_split

all_X = train[columns]
all_y = train['Survived']

train_X, test_X, train_y, test_y = train_test_split(
    all_X, all_y, test_size=0.20,random_state=0)
from sklearn.metrics import accuracy_score
lr = LogisticRegression()
lr.fit(train_X, train_y)
predictions = lr.predict(test_X)
accuracy = accuracy_score(test_y, predictions)

print(accuracy)
from sklearn.model_selection import cross_val_score

lr = LogisticRegression()
scores = cross_val_score(lr, all_X, all_y, cv=10)
scores.sort()
accuracy = scores.mean()

print(scores)
print(accuracy)
lr = LogisticRegression()
lr.fit(all_X,all_y)
holdout_predictions = lr.predict(holdout[columns])
holdout_ids = holdout["PassengerId"]
submission_df = {"PassengerId": holdout_ids,
                 "Survived": holdout_predictions}
submission = pd.DataFrame(submission_df)
submission.head()
submission.to_csv('submission.csv', index=False)