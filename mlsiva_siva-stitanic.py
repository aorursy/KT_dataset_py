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
import pandas as pd
inputds=pd.read_csv('../input/train.csv')
inputds.fillna(inputds.mean(),inplace=True)
inputds.fillna('S',inplace=True)
#removing the column Cabin as it has more than 60% Null values
inputds=inputds.drop('Cabin',axis=1)
inputds=inputds.drop('Name',axis=1)
inputds[inputds['PassengerId']==62]
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
#we can use one hot encoding when we want to encode for multiple columns at once
#but One hot encoding wont work when the in case of many categories in a same column(Eg:happy, extremely happy, okay,poor)
inputds['Sex']=le.fit_transform(inputds['Sex'])
inputds['Embarked']=le.fit_transform(inputds['Embarked'])
inputds['Ticket']=le.fit_transform(inputds['Ticket'])
from sklearn.ensemble import ExtraTreesClassifier
etc=ExtraTreesClassifier()
y=inputds['Survived'].tolist()
inputds1=inputds
inputds2=inputds1.drop('Survived',axis=1)

x=inputds2.values
etc.fit(x,y)
#inputds2.head
etc.feature_importances_
inputds2.columns
x1=inputds2.drop('Sex',axis=1).drop('Age',axis=1).drop('SibSp',axis=1).drop('Parch',axis=1).drop('Embarked',axis=1).values

# have add some code to visualize the feature selection output in a graph
from sklearn.naive_bayes import MultinomialNB
mnb=MultinomialNB()
mnb.fit(inputds2,y)
testds=pd.read_csv('../input/test.csv')
testds.fillna(testds.mean(),inplace=True)
testds.fillna('S',inplace=True)

testds=testds.drop('Cabin',axis=1)

testds=testds.drop('Name',axis=1)
testds['Sex']=le.fit_transform(testds['Sex'])

inputds[inputds['PassengerId']==62]

testds['Embarked']=le.fit_transform(testds['Embarked'])

testds['Ticket']=le.fit_transform(testds['Ticket'])

#testds['Name']=le.fit_transform(testds['Name'])
testds
predictt=mnb.predict(testds)
finalseries=pd.Series(predictt)
finalseries.to_csv('finalprediction.csv')
round( mnb.score(x,y)* 100, 2)
y
testrst=pd.read_csv('../input/gender_submission.csv')
testrst1=testrst['Survived']
from sklearn.metrics import accuracy_score
accuracy_score(testrst1,predictt)
#inputds.select('PassengerId')
for i in predictt:
    print(predictt[i])
testd=testds.PassengerId.tolist()
testrst1=testrst.Survived.tolist()
#spam_df.head(10).text.tolist() + spam_df.tail(10).text.tolist()
i=0
for a,b in zip(testd,testrst1):
    print(str(testd[i][:50])+" actual value is ("+str(testrst1[i]) +") predicted value is "+str(predictt[i]))
    i+=1
from sklearn.metrics import accuracy_score
accuracy_score(y, predictt)
finalseries













































inputds.select[PassengerId]

















