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
train_address="/kaggle/input/titanic/train.csv"

test_address="/kaggle/input/titanic/test.csv"

train_set=pd.read_csv(train_address)

test_set=pd.read_csv(test_address)
features=['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']

X=train_set[features]

y=train_set.Survived

X_test=test_set[features]

print(X_test.info())

print(X.info())

X.head()
#Label Encoder

from sklearn.ensemble import RandomForestRegressor

from sklearn import preprocessing

encoder=preprocessing.LabelEncoder()

X_le=X.copy()

X_le['Embarked']=X_le['Embarked'].astype(str)

X_le['Embarked']=encoder.fit_transform(X_le['Embarked'])

X_le['Sex']=encoder.fit_transform(X_le['Sex'])

X_test_le=X_test.copy()

X_test_le['Embarked']=X_test_le['Embarked'].astype(str)

X_test_le['Embarked']=encoder.fit_transform(X_test_le['Embarked'])

X_test_le['Sex']=encoder.fit_transform(X_test_le['Sex'])

X_le.info()
#One Hot Encoder

from sklearn import preprocessing

ohencoder=preprocessing.OneHotEncoder(handle_unknown='ignore', sparse=False)

X_oh=train_set[features]

X_test_oh=test_set[features]

X_oh['Embarked']=X_oh['Embarked'].astype(str)

X_test_oh['Embarked']=X_test_oh['Embarked'].astype(str)

cols=(X_oh.dtypes=='object')

cols=list(cols[cols].index)

#Apply one-hot encoder to each column with categorical data

X_oh=pd.DataFrame(ohencoder.fit_transform(X_oh[cols]))

X_test_oh=pd.DataFrame(ohencoder.transform(X_test_oh[cols]))

#getting back indexes

X_oh.index=train_set.index

X_test_oh.index=test_set.index



X_oh = pd.concat([train_set[features].drop(cols,axis=1), X_oh],axis=1)

X_test_oh = pd.concat([test_set[features].drop(cols,axis=1), X_test_oh],axis=1)

X_oh.isnull().any()
#using classifier

feat=['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']

X_c=X_le.copy()

X_age=X_c[~X_c['Age'].isnull()][feat]

X_age.dropna(inplace=True)

y_age=X_age.Age

X_age=X_age.drop(['Age','Embarked'],axis=1)

from sklearn.ensemble import RandomForestRegressor

rf=RandomForestRegressor(n_estimators=100, max_depth=5, random_state=1)

rf.fit(X_age,y_age)

X_age_test=X_c[X_c['Age'].isnull()][X_age.columns]

y_age_out=pd.DataFrame(rf.predict(X_age_test))

y_age_out.index=X_age_test.index

for i in y_age_out.index:

    X_c.loc[i,'Age']=y_age_out.loc[i,0]

X_c.dropna(axis=1,inplace=True)

X_c.info()
X_c_test=X_test_le.copy()

X_age_t=X_c_test[~X_c_test['Age'].isnull()][feat]

X_age_t.dropna(inplace=True)

y_age_t=X_age_t.Age

X_age_t=X_age_t.drop(['Age','Embarked'],axis=1)

from sklearn.ensemble import RandomForestRegressor

rft=RandomForestRegressor(n_estimators=100, max_depth=5, random_state=1)

rft.fit(X_age_t,y_age_t)

X_test_age_test=X_c_test[X_c_test['Age'].isnull()][X_age_t.columns]

y_test_age_out=pd.DataFrame(rft.predict(X_test_age_test))

y_test_age_out.index=X_test_age_test.index

for i in y_test_age_out.index:

    X_c_test.loc[i,'Age']=y_test_age_out.loc[i,0]

X_c_test['Fare']=X_c_test['Fare'].fillna(np.mean(X_c_test['Fare']))

X_c_test.info()
from sklearn.impute import SimpleImputer

imp=SimpleImputer()

columns=X_oh.columns

X_oh=pd.DataFrame(imp.fit_transform(X_oh))

X_oh.columns=columns

X_test_oh=pd.DataFrame(imp.fit_transform(X_test_oh))

X_test_oh.columns=columns

from sklearn.impute import SimpleImputer

imp=SimpleImputer()

X_le=pd.DataFrame(imp.fit_transform(X_le))

X_le.columns=features

X_test_le=pd.DataFrame(imp.fit_transform(X_test_le))

X_test_le.columns=features

print(X_le.info())

print(X_test_le.info())

X_le.head()
from xgboost import XGBClassifier

xgb=XGBClassifier()
model_1 = RandomForestRegressor(n_estimators=50, random_state=0)

model_2 = RandomForestRegressor(n_estimators=100, random_state=0)

model_3 = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0)

model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)

model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)
from sklearn import model_selection

from sklearn.ensemble import RandomForestRegressor

x1,x2,y1,y2=model_selection.train_test_split(X_c,y)

rf=RandomForestRegressor(n_estimators=100, max_depth=5, random_state=1)

rf.fit(x1,y1)

pred=rf.predict(x2)

np.round(pred)

np.mean(np.round(pred)==y2)*100
from sklearn import model_selection

x1,x2,y1,y2=model_selection.train_test_split(X_oh,y)

rf=RandomForestRegressor(n_estimators=100, max_depth=5, random_state=1)

rf.fit(x1,y1)

pred=rf.predict(x2)

np.round(pred)

np.mean(np.round(pred)==y2)*100
from sklearn import model_selection

x1,x2,y1,y2=model_selection.train_test_split(X_le,y)

rf=RandomForestRegressor(n_estimators=100, max_depth=5, random_state=1)

rf.fit(x1,y1)

pred=rf.predict(x2)

np.round(pred)

np.mean(np.round(pred)==y2)*100

#predictions=rf.predict(X_test)

#predictions
#Implementing the Random Forest Model

rf.fit(X_oh,y)

predictions=np.round(rf.predict(X_test_oh)).astype('int')

output = pd.DataFrame({'PassengerId': test_set.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)
