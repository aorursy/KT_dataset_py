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
# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
# reading in data
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
train.info()
test.info()
train.describe()
train.describe(include=['O'])
train[['Pclass','Survived']].groupby(['Pclass'], as_index=False)['Survived'].agg(['mean','count'])
train.Sex=pd.Categorical(train.Sex)
train.Sex=train.Sex.cat.codes
print(train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))
test.Sex=pd.Categorical(test.Sex)
test.Sex=test.Sex.cat.codes
#Survival rate for people without cabin info is significantly lower than the one with info. Therefore, null data matters..
train.Cabin=train.Cabin.fillna('N').str[:1]
print(train[["Cabin", "Survived"]].groupby(['Cabin'], as_index=False)['Survived'].agg(['mean','count']))
#Cabin section is found to be related to Pclass. It looks like people with cabin information has significantly better survival rate.
#Therefore, I will only categorize by existence of cabin info.
train.Cabin=train.Cabin.fillna('N').str[:1]
train['Cabin_info']=1
train['Cabin_info'][train.Cabin=='N']=0
print(train[["Cabin","Pclass","Survived"]].groupby(["Cabin","Pclass"], as_index=False)['Survived'].agg(['mean','count']))
print(train[["Cabin_info","Pclass","Survived"]].groupby(["Pclass","Cabin_info"], as_index=False)['Survived'].agg(['mean','count']))
test.Cabin=test.Cabin.fillna('N').str[:1]
test['Cabin_info']=1
test['Cabin_info'][test.Cabin=='N']=0
#I choose not to use median for null values because it looked like survival rate is significantly lower for null.
train['Age_group']=pd.cut(train.Age, 8).cat.codes
test['Age_group']=pd.cut(test.Age, 8).cat.codes
train[["Age_group", "Survived"]].groupby(['Age_group'], as_index=False)['Survived'].agg(['mean','count'])
#As you can imagine, if we know family size, parch, sibsp, and age, we, humans, can give a good guess if the person is parent/child/alone.
#If neural network does things right, one of the nodes should give family size, and use it for estimation of survival rate. However, there is no
#guarantee that neural net would come up with that. As a result, I chose to add the variable manually. As sample is too small to be significant 
#for 8-member and 11-member family, I chose to merge them to 7.
train['Family_size']=train['Parch']+train['SibSp']+1
train.Family_size[train.Family_size>6]=7
test['Family_size']=test['Parch']+test['SibSp']+1
test.Family_size[test.Family_size>6]=7
print(train[["Family_size","Survived"]].groupby(["Family_size"], as_index=False)['Survived'].agg(['mean','count']))
#So far, I have considered having no information as something that matters. However, in this case, there are only 2 nulls. As a result, 
#it is hard to say that null info is significant. Therefore, I chose to make null info to one of the majorities.
train.Embarked=pd.Categorical(train.Embarked)
train.Embarked=train.Embarked.cat.codes
train.Embarked[train.Embarked==-1]=2
test.Embarked=pd.Categorical(test.Embarked)
test.Embarked=test.Embarked.cat.codes
test.Embarked[test.Embarked==-1]=2
train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False)['Survived'].agg(['mean','count'])
#Fare is definitely related to Pclass, but relationship is not linear. Additionally, there is no clear cut in Fare to distinguish class.
#However, there is one meaningful observation: expensive first class is more likely to survive. This was not necessarily the case for 2nd and 3rd.
#As there is significant survival difference from $50, I chose to give category for that.
train['Fare_cut']=pd.cut(train.Fare, 31)
test['Fare_cut']=pd.cut(test.Fare, 31)
train[["Fare_cut","Pclass","Survived"]].groupby(["Fare_cut","Pclass"], as_index=False)['Survived'].agg(['mean','count'])
#It looks like fare affects survival rates in a different way by class. This is another reason why neural network would make more sense.
train['Fare_cat']=0
train['Fare_cat'][train.Fare>50]=1
test['Fare_cat']=0
test['Fare_cat'][test.Fare>50]=1
train[["Fare_cat","Pclass","Survived"]].groupby(["Fare_cat","Pclass"], as_index=False)['Survived'].agg(['mean','count'])
#There are many different kinds of Prefix, but only few of them have enough sample. I chose to take Rev and Dr separately, but it's your choice.
train['Prefix']=train.Name.str.replace('(.*, )|(\\..*)', '')
for i in np.arange(len(train.Prefix)):
    if train.Prefix[i] not in ('Miss','Mrs','Master','Mr','Rev','Dr'):
        train.Prefix[i]='Unique'
train.Prefix=pd.Categorical(train.Prefix)
train.Prefix=train.Prefix.cat.codes

test['Prefix']=test.Name.str.replace('(.*, )|(\\..*)', '')
for i in np.arange(len(test.Prefix)):
    if test.Prefix[i] not in ('Miss','Mrs','Master','Mr','Rev','Dr'):
        test.Prefix[i]='Unique'
test.Prefix=pd.Categorical(test.Prefix)
test.Prefix=test.Prefix.cat.codes

train[["Prefix","Sex","Survived"]].groupby(["Sex","Prefix"], as_index=False)['Survived'].agg(['mean','count'])
del train['Name']
del train['PassengerId']
del train['Age']
del train['Ticket']
del train['Fare']
del train['Cabin']
del train['Fare_cut']

del test['Name']
del test['Age']
del test['Ticket']
del test['Fare']
del test['Cabin']
del test['Fare_cut']
print(train.info())
print(test.info())
from sklearn.cross_validation import train_test_split
X_all = train.drop(['Survived'], axis=1)
y_all = train['Survived']

num_test = 200
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=23)
# neural network
#For choosing the number of neurons, I chose to test all numbers below 22 which is double the number of all variables.
#After checking results, I chose to use 7 neurons for double layer.
from sklearn.neural_network import MLPClassifier
neural_record=pd.DataFrame(columns=['Single_In','Double_In','Single_Out','Double_Out','Single_Net','Double_Net'],index=np.arange(21))
for i in np.arange(21):
    c = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(i+1), random_state=1,activation='logistic',max_iter=3000)
    c.fit(X_train, y_train)
    neural_record['Single_In'][i]=c.score(X_train,y_train) #In-sample result with single layer
    neural_record['Single_Out'][i]=np.mean(c.predict(X_test)==y_test) #Out-of-sample result with single layer
    neural_record['Single_Net'][i]=c
    c2 = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(i+1,i+1), random_state=1,activation='logistic',max_iter=3000)
    c2.fit(X_train, y_train)
    neural_record['Double_In'][i]=c2.score(X_train,y_train) #In-sample result with double layer
    neural_record['Double_Out'][i]=np.mean(c2.predict(X_test)==y_test) #Out-of-sample result with double layer
    neural_record['Double_Net'][i]=c2
    print(i,c.score(X_train,y_train),np.mean(c.predict(X_test)==y_test),c2.score(X_train,y_train),np.mean(c2.predict(X_test)==y_test))
ids = test['PassengerId']
predictions = neural_record['Single_Net'][10].predict(test.drop('PassengerId', axis=1))
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
output.to_csv('titanic-predictions-SP.csv', index = False)
predictions = neural_record['Double_Net'][6].predict(test.drop('PassengerId', axis=1))
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
output.to_csv('titanic-predictions-SP-double-hidden-layer.csv', index = False)