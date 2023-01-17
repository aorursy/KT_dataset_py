# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



traindata = pd.read_csv("../input/train.csv")

testdata  = pd.read_csv("../input/test.csv")
traindata.info()

testdata.info()
traindata.head()
value = traindata['Age'].mean()

traindata['Age'].fillna(value,inplace=1)

traindata.info()

traindata['Embarked'].value_counts()
traindata['Embarked'].fillna(value='S',inplace=1)

traindata.info()
del traindata['PassengerId']

del traindata['Cabin']

del traindata['Name']

del traindata['Ticket']

del testdata['Cabin']

del testdata['Name']

del testdata['Embarked']

del testdata['Ticket']
traindata.info()
col_list = ['Pclass','Sex','Age','SibSp','Parch','Fare']

Y_train = traindata['Survived']

Y_train.describe()

traindata = traindata[col_list]

traindata.info()

Y_train.describe()
def sexConv(g):

    if g == 'male':

        return 1

    return 0

traindata['Sex'] = traindata['Sex'].apply(sexConv)

traindata.head()
from sklearn.linear_model import LogisticRegression

logModel = LogisticRegression()

logModel.fit(traindata,Y_train)
testdata['Sex'] = testdata['Sex'].apply(sexConv)

val = testdata['Age'].mean()

testdata['Age'].fillna(value=val,inplace=1)

testdata['Fare'].fillna(0,inplace=1)

testdatapid = testdata['PassengerId']

testdata = testdata.ix[:,testdata.columns != "PassengerId"]
predictions = logModel.predict(testdata)
final = pd.DataFrame(testdatapid,columns=['PassengerId'])

final['Survived'] = predictions

final
final.to_csv('output',index=False)
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))