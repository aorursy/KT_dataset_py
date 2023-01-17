# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.preprocessing import LabelEncoder



#Importing the data

train=pd.read_csv('/kaggle/input/titanic/train.csv')

test=pd.read_csv('/kaggle/input/titanic/test.csv')



#checking the data to find null values

train.info(),test.info()
train.head()
test.head()
train['Sex']=(train['Sex']=='male').astype('int')

test['Sex']=(test['Sex']=='male').astype('int')



model=LabelEncoder().fit(train['Embarked'].tolist())

train['Embarked']=model.transform(train['Embarked'].tolist())

test['Embarked']=model.transform(test['Embarked'].tolist())

#Removing the Sibsp ,parch and merging it into a single column as family

train['Family']=train['SibSp']+train['Parch']

test['Family']=test['SibSp']+test['Parch']

end=test['PassengerId']



#we can remove data which is not needed for both the data

train.drop(['PassengerId','Name','Ticket','Cabin','Parch','SibSp'],inplace=True,axis=1)

test.drop(['PassengerId','Name','Ticket','Cabin','SibSp','Parch'],inplace=True,axis=1)

y=train['Survived']

train.drop(['Survived'],inplace=True,axis=1)
train.head()
test.head()
check=train.groupby(['Pclass','Sex'])['Age'].apply(np.mean).to_frame()

check
for i in train[train['Age'].isna()].index:

  train.loc[i,'Age']=int(check.loc[train.iloc[i]['Pclass'],train.iloc[i]['Sex']])



for i in test[test['Age'].isna()].index:

  test.loc[i,'Age']=int(check.loc[test.iloc[i]['Pclass'],test.iloc[i]['Sex']])
train.groupby(['Embarked','Pclass'])['Fare'].apply(np.mean)
test.loc[test['Fare'].isna(),'Fare']=14.644

test.isna().sum()
waste,cuts=pd.cut(train['Age'],5,retbins=True)

cuts[0]=0



train['Age']=pd.cut(train['Age'],cuts)

test['Age']=pd.cut(test['Age'],cuts)

model=LabelEncoder().fit(train['Age'])

train['Age']=model.transform(train['Age'])

test['Age']=model.transform(test['Age'])
train.head()
test.head()
from sklearn.linear_model import LogisticRegression

model=LogisticRegression().fit(train,y)

model.score(train,y)
from sklearn.svm import SVC

model=SVC(kernel='linear').fit(train,y)

model.score(train,y)
from sklearn.ensemble import GradientBoostingClassifier

model=GradientBoostingClassifier(learning_rate=0.9,max_depth=4).fit(train,y)

model.score(train,y)
from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier(random_state=0).fit(train,y)

model.score(train,y)
from sklearn.tree import DecisionTreeClassifier

model=DecisionTreeClassifier(random_state=0).fit(train,y)

model.score(train,y)
result=pd.DataFrame(model.predict(test),index=end,columns=['Survived'])
result.to_csv('result.csv')