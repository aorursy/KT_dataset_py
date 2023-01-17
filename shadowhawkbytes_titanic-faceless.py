import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.cross_validation import train_test_split

from sklearn.naive_bayes import GaussianNB 

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
train=pd.read_csv("../input/train.csv",header=0)
def maleorfe(var):

    if var=='female':

        return 0

    else:

        return 1
train['Sex']=train['Sex'].apply(lambda x:maleorfe(x))
df=train[['PassengerId','Pclass','Parch','SibSp','Age','Sex','Survived']]
df.dropna(axis=0,how='any',inplace=True)
X=df[['PassengerId', 'Pclass', 'Age','Sex']]

y=df['Survived']
xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.25)
clf=LogisticRegression()

clf.fit(xtrain,ytrain)

pred=clf.predict(xtest)

acc=accuracy_score(pred,ytest)
acc
test=pd.read_csv("../input/test.csv",header=0)
test['Sex']=test['Sex'].apply(lambda x:maleorfe(x))
x_test=test[['PassengerId', 'Pclass', 'Age', 'Sex']]
x_test['Age'].fillna(0, inplace=True)
pr=clf.predict(x_test).astype(int)
ans = pd.DataFrame({'PassengerId':x_test.PassengerId, 'Survived':pr})
ans.to_csv('finalanswer.csv', index=False)
import csv



with open('finalanswer.csv') as csvfile:

    readCSV = csv.reader(csvfile, delimiter=',')

    for row in readCSV:

        print(row[0],row[1],sep=',')