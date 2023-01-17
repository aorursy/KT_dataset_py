# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train=pd.read_csv('../input/titanic/train.csv')
train.head()
test  = pd.read_csv('../input/titanic/test.csv')

test.head()
full=train.append(test,ignore_index=True)

full.head()
full.describe()
full.info()
full['Age']=full['Age'].fillna(full['Age'].mean())

full['Fare']=full['Fare'].fillna(full['Fare'].mean())

full.info()
full.head()
full['Embarked'].value_counts()
full['Embarked'] = full['Embarked'].fillna( 'S' )
full['Cabin'].head()
full['Cabin']=full['Cabin'].fillna('U')

full['Cabin'].head()
full.info()
sex_mapDict={'male':1,

            'female':0}

full['Sex']=full['Sex'].map(sex_mapDict)

full.head()
embarkedDf=pd.DataFrame()



embarkedDf=pd.get_dummies(full['Embarked'],prefix='Embarked')

embarkedDf.head()
full=pd.concat([full,embarkedDf],axis=1)

full.head()
full.drop('Embarked',axis=1,inplace=True)

full.head()
pclassDf=pd.DataFrame()

pclassDf=pd.get_dummies(full['Pclass'],prefix='Pclass')

pclassDf.head()
full=pd.concat([full,pclassDf],axis=1)

full.head()
full.drop('Pclass',axis=1,inplace=True)

full.head()
def getTitle(name):

    str1=name.split( ',' )[1] #Mr. Owen Harris

    str2=str1.split( '.' )[0]#Mr

    #strip() 方法用于移除字符串头尾指定的字符（默认为空格）

    str3=str2.strip()

    return str3
titleDf=pd.DataFrame()

titleDf['Title'] = full['Name'].map(getTitle)

titleDf['Title'].value_counts()
title_mapDict = {

                    "Capt":       "Officer",

                    "Col":        "Officer",

                    "Major":      "Officer",

                    "Jonkheer":   "Royalty",

                    "Don":        "Royalty",

                    "Sir" :       "Royalty",

                    "Dr":         "Officer",

                    "Rev":        "Officer",

                    "the Countess":"Royalty",

                    "Dona":       "Royalty",

                    "Mme":        "Mrs",

                    "Mlle":       "Miss",

                    "Ms":         "Mrs",

                    "Mr" :        "Mr",

                    "Mrs" :       "Mrs",

                    "Miss" :      "Miss",

                    "Master" :    "Master",

                    "Lady" :      "Royalty"

                    }

titleDf['Title'] = titleDf['Title'].map(title_mapDict)
titleDf = pd.get_dummies(titleDf['Title'])

titleDf.head()
full=pd.concat([full,titleDf],axis=1)

full.head(1)
full.drop('Name',axis=1,inplace=True)

full.head(1)
full.info()
cabinDf = pd.DataFrame()

full['Cabin']=full['Cabin'].map(lambda c:c[0])
cabinDf=pd.get_dummies(full['Cabin'],prefix='Cabin')

cabinDf.head()
full=pd.concat([full,cabinDf],axis=1)

full.head(1)
full.drop('Cabin',axis=1,inplace=True)

full.head(1)
familyDf=pd.DataFrame()

familyDf[ 'FamilySize' ] = full[ 'Parch' ] + full[ 'SibSp' ] + 1

familyDf[ 'Family_Single' ] = familyDf[ 'FamilySize' ].map( lambda s : 1 if s == 1 else 0 )

familyDf[ 'Family_Small' ]  = familyDf[ 'FamilySize' ].map( lambda s : 1 if 2 <= s <= 4 else 0 )

familyDf[ 'Family_Large' ]  = familyDf[ 'FamilySize' ].map( lambda s : 1 if 5 <= s else 0 )

familyDf.head()
full = pd.concat([full,familyDf],axis=1)

full.head()
corrDf = full.corr() 

corrDf['Survived'].sort_values(ascending =False)
full_X = pd.concat( [titleDf,

                     pclassDf,

                     familyDf,

                     full['Fare'],

                     cabinDf,

                     embarkedDf,

                     full['Sex']

                    ] , axis=1 )

full_X.head()
sourceRow=891

source_X = full_X.loc[0:sourceRow-1,:]

source_y = full.loc[0:sourceRow-1,'Survived']

pred_X = full_X.loc[sourceRow:,:]

pred_X = full_X.loc[sourceRow:,:]
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(source_X,

                                             source_y,train_size=.8 )
print('原始数据特征：',source_X.shape,

    '训练数据特征：', X_train.shape,

     '测试数据特征：',X_test.shape)



print('原始数据标签：',source_y.shape,

    '训练数据标签：', y_train.shape,

     '测试数据标签：',y_test.shape)
from sklearn.linear_model import LogisticRegression

model=LogisticRegression()

model.fit(X_train,y_train)
model.score(X_test,y_test)
pred_Y = model.predict(pred_X)

pred_Y=pred_Y.astype(int)
passenger_id = full.loc[sourceRow:,'PassengerId']

predDf = pd.DataFrame( 

    { 'PassengerId': passenger_id , 

     'Survived': pred_Y } )

predDf.shape
predDf.head()
predDf.to_csv()