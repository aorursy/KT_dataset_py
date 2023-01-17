# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import xgboost as xgb #xgboost

from sklearn.preprocessing import LabelEncoder #encoding string

from sklearn.model_selection import KFold, GridSearchCV # cv and parameter search

from sklearn.metrics import confusion_matrix, accuracy_score

import pandas as pd



rng=np.random.RandomState(1)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.


gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test_data = pd.read_csv("../input/titanic/test.csv")

train_data = pd.read_csv("../input/titanic/train.csv")

# train_data.head()
# test_data.head()
##features used in training

selefeature=['Pclass','Sex','Age','SibSp','Parch','Fare']

predfeature=['Survived']

tempx=train_data[selefeature].append(test_data[selefeature])

allx=tempx.copy()

# check nan value

# type(allx)

# allx.isnull().sum()

# np.flatnonzero(allx['Fare'].isnull()==True)

# use median in train and test data set to replace nan

agenanind=allx['Age'].isnull()

allx.loc[agenanind,'Age']=allx.loc[~agenanind,'Age'].median()

# deal with the only nan in Fare

farenanind=allx['Fare'].isnull()

allx.loc[farenanind,'Fare']=allx.loc[~farenanind,'Fare'].median()

#columne to add to for nan in age

allx['agenonexist']=agenanind.astype(int)

##encoder sex

le=LabelEncoder()

allx['Sex']=le.fit_transform(allx['Sex'])

# allx.head()
## data formulation

indcut=train_data.shape[0]

alllen=allx.shape[0]

trainvalid_x=allx.iloc[0:indcut].values.copy()

test_x=allx.iloc[indcut:alllen].values.copy()

trainvalid_y=train_data[predfeature].values.copy()
# # parameter searching? 6 100

# gbm=xgb.XGBClassifier()

# clf = GridSearchCV(gbm,{'max_depth': [2,4,6],'n_estimators': [50,100,200,500]},verbose=1)

# clf.fit(trainvalid_x,trainvalid_y[:,0])

# print(clf.best_score_)

# print(clf.best_params_)
#crosss validation training 

kf=KFold(n_splits=10,shuffle=True,random_state=rng)

for train_index,valid_index in kf.split(trainvalid_x):

    gbm=xgb.XGBClassifier(max_depth=6,n_estimators=100).fit(trainvalid_x[train_index],trainvalid_y[train_index,0])

    predictions=gbm.predict(trainvalid_x[valid_index])

    actuals=trainvalid_y[valid_index,0]

#     print(confusion_matrix(actuals,predictions))

    print(accuracy_score(actuals,predictions))
#trainign on whole train data set

gbm=xgb.XGBClassifier(max_depth=6,n_estimators=100).fit(trainvalid_x,trainvalid_y[:,0])

predictions=gbm.predict(test_x)
submission=pd.DataFrame({ 'PassengerId': test_data['PassengerId'],

                            'Survived': predictions})

submission.to_csv("submission.csv",index=False)