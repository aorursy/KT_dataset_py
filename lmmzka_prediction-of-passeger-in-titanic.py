# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("../input/train.csv", header=None)
train_data = train_data.rename(columns=train_data.iloc[0])

train_data.drop([0],inplace = True)

train_data.drop(['Name','Parch','Cabin','SibSp','Ticket','Embarked'],axis = 1,inplace = True)

train_data['Sex'] = train_data['Sex'].replace('female',1.0)

train_data['Sex'] = train_data['Sex'].replace('male',0.0)

train_data.iloc[:,2:] = train_data.astype(float)

train_data['Pclass'] = (1/train_data['Pclass']-(1/3))/(2/3)

train_data['Fare'] = (train_data['Fare']-train_data['Fare'].min())/(train_data['Fare'].max()-train_data['Fare'].min())

train_data['Age'] = abs((train_data['Age']-train_data['Age'].mean())/(train_data['Age'].max()-train_data['Age'].min()))
train_data.apply(lambda x: sum(x.isnull()),axis=0) 
train_data['Age'].fillna(0, inplace=True)
passLabel = train_data['Survived']

train_data.drop(['Survived'],axis = 1,inplace = True)

train_data.drop(['PassengerId'],axis = 1,inplace = True)
test_data = pd.read_csv("../input/test.csv", header=None)
test_data = test_data.rename(columns=test_data.iloc[0])

test_data.drop([0],inplace = True)

test_data.drop(['Name','Parch','Cabin','SibSp','Ticket','Embarked'],axis = 1,inplace = True)

test_data['Sex'] = test_data['Sex'].replace('female',1.0)

test_data['Sex'] = test_data['Sex'].replace('male',0.0)

test_data.iloc[:,1:] = test_data.astype(float)

test_data['Pclass'] = (1/test_data['Pclass']-(1/3))/(2/3)

test_data['Fare'] = (test_data['Fare']-test_data['Fare'].min())/(test_data['Fare'].max()-test_data['Fare'].min())

test_data['Age'] = abs((test_data['Age']-test_data['Age'].mean())/(test_data['Age'].max()-test_data['Age'].min()))

test_data['Age'].fillna(0, inplace=True)

test_data.drop(['PassengerId'],axis = 1,inplace = True)
from sklearn.naive_bayes import BernoulliNB

clf = BernoulliNB()

clf.fit(train_data, passLabel)

BernoulliNB()
test_data['Fare'].fillna(0, inplace=True)  

predict=clf.predict(test_data)
data = pd.read_csv("../input/genderclassmodel.csv", header=None)
count = 0

for i in range(predict.shape[0]):

    if(predict[i]==data.iloc[:,1][i+1]):

        count = count + 1

    else:

        pass

accuracy = count/predict.shape[0]

print(accuracy)
sub_data = pd.DataFrame(predict)
sub_data.index = sub_data.index+892

sub_data.index.name = 'PassengerId'
sub_data.columns = ['Survived']
sub_data.head()
sub_data.to_csv("submission.csv")