# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
print(train.keys())
#raw data

fig, ax = plt.subplots(1,3)

ax[0].hist(train.Age,rwidth=0.9)

ax[0].set_title('Age')

ax[1].hist(train.Sex,rwidth=0.9)

ax[1].set_title('Sex')

ax[2].hist(train.Survived,rwidth=0.9)

ax[2].set_title('Survived')

fig.set_figwidth(15)

plt.show()
fig, ax = plt.subplots(1,2)

#how likely are you to survive based on gender?

ax[0].hist([train.Sex[train.Survived==1], train.Sex[train.Survived==0]],rwidth=0.5)

ax[0].legend(labels=['Survived','Died'])

ax[0].set_title('Sex')

#how likely are you to survive based on age?

ax[1].hist([train.Age[train.Survived==1], train.Age[train.Survived==0]])

ax[1].legend(labels=['Survived','Died'])

ax[1].set_title('Age')

fig.set_figwidth(15)

plt.show()
#Combining these two...

fig, (ax1, ax2) = plt.subplots(1,2)

ax1.hist([train.Age[(train.Survived==1) & (train.Sex=='male')],

          train.Age[(train.Survived==0) & (train.Sex=='male')]],rwidth=0.9)

ax1.legend(labels=['Survived','Died'])

ax1.set_title('Male')

ax2.hist([train.Age[(train.Survived==1) & (train.Sex=='female')],

          train.Age[(train.Survived==0) & (train.Sex=='female')]],rwidth=0.9)

ax2.legend(labels=['Survived','Died'])

ax2.set_title('Female')

fig.set_figwidth(15)

plt.show()

#Deaths based on ticket class...

plt.hist([train.Pclass[train.Survived==1],train.Pclass[train.Survived==0]])

plt.xticks([1,2,3])

plt.legend(['Survived','Died'])

plt.show()
import missingno as msno

#missing data in train.csv

msno.bar(train,color='lightblue')

plt.show()
#missing data in test.csv

msno.bar(test,color='lightblue')

plt.show()
def clean_data(data):

    #Replace missing ages with median age

    data.Age.fillna(data.Age.median(),inplace=True)



    #fill Cabin=Nan with 'unknown'

    data.Cabin.fillna('unknown',inplace=True)



    #replace embarked with mode

    data.Embarked.fillna(data.Embarked.mode()[0],inplace=True)

    

    #replace missing fare with median fare

    data.Fare.fillna(data.Fare.median(),inplace=True)

    

    return data



train = clean_data(train)

test = clean_data(test)
#train data

msno.bar(train,color='lightblue')

plt.show()
#test data

msno.bar(test,color='lightblue')

plt.show()
train.head()
#feature engineering

def reformat_data(data):

    #make gender numerical

    data.Sex = data.Sex.replace('male',0)

    data.Sex = data.Sex.replace('female',1)

    

    #make embarked numerical

    data.Embarked = data.Embarked.map({'S':0,'Q':1,'C':2,'U':3})

    

    #create family_size and alone column

    data['FamilySize'] = data.SibSp + data.Parch + 1

    data['Alone'] = 1

    data.Alone[data.FamilySize>1] = 0

    

    #create title column, Mr, Mrs etc...

    titles = [name.split(' ')[1] for name in data.Name]

    data['Title'] = titles

    data.Title = data.Title.map({'Mr.':1,'Miss.':2,'Mrs.':3,'Master.':4})

    data.Title.fillna(5, inplace=True)



    return data

    

train = reformat_data(train)
train.Title.value_counts()
#final check 

msno.bar(train,color='lightblue')

plt.show()
train.head()
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split



#features I want to use in my model

features = ['Pclass','Sex','Age','FamilySize','Alone','Fare','Embarked','Title']



train_df, test_df = train_test_split(train, test_size=0.25)

X_train = train_df[features]

Y_train = train_df['Survived']

X_test = test_df[features]

Y_test = test_df['Survived']
#train the model

rf = RandomForestClassifier(n_estimators=100)

rf.fit(X_train, Y_train)

preds = rf.predict(X_test)

print('n_estimators: %i' %100)

print('train_score: %.2f ' %(rf.score(X_train, Y_train)*100))

print('test_score: %.2f ' %(rf.score(X_test, Y_test)*100))

print('')
test = pd.read_csv('/kaggle/input/titanic/test.csv')

test = clean_data(test)

test = reformat_data(test)

preds = rf.predict(test[features])

test['Survived'] = preds

print('test.csv predictions')

print(preds)
submission = test[['PassengerId','Survived']]

submission.head()
submission.to_csv('submission.csv',index=False)