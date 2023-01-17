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
ds=pd.read_csv('/kaggle/input/titanic/train.csv')
dt=pd.read_csv('/kaggle/input/titanic/test.csv')
ds.columns
ds.isna().sum()
dt.isna().sum()
ds.drop(['Cabin','Name','Ticket','Embarked'],axis=1,inplace=True)
dt.drop(['Cabin','Name','Ticket','Embarked'],axis=1,inplace=True)
ds['Age'].fillna(value=0,inplace=True)
ds.isna().sum()
ds["Age"]=(ds["Age"]-29.699118)/(80.000000- 0.420000)
dt['Age'].fillna(value=0,inplace=True)

dt['Fare'].fillna(value=0,inplace=True)
dt.isna().sum()
dt['Age'].describe()
dt["Age"]=(dt["Age"]-30.272590)/(76.000000- 0.170000)
ds['Fare'].describe()
ds['Fare']=(ds['Fare']- 32.204208)/(512.329200)
dt['Fare'].describe()
dt['Fare']=(dt['Fare']- 35.541956)/(512.329200)
dt.isna().sum()
gendermap={

    'male':0,

    'female':1

}
ds['Sex']=ds['Sex'].map(gendermap)
dt['Sex']=dt['Sex'].map(gendermap)
from sklearn.model_selection import train_test_split

x=ds.drop(['Survived','PassengerId'],axis=1)

y=ds['Survived']

xtrain,xval,ytrain,yval=train_test_split(x,y,random_state=42)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver='liblinear',multi_class='ovr')

lr.fit(xtrain,ytrain)
lr.score(xval,yval)
predict=lr.predict(dt.drop(['PassengerId'],axis=1))
submission=pd.DataFrame(data=predict,columns=['Survived'])
submission['PassengerId']=dt['PassengerId']
# df=pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
# submission=submission[df.columns]
submission.set_index('PassengerId',inplace=True)
submission.head()
submission.to_csv('gender_submission.csv')