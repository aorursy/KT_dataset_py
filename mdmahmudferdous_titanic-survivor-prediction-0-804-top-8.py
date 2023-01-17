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

import warnings

warnings.filterwarnings('ignore')
train=pd.read_csv('/kaggle/input/titanic/train.csv')

test=pd.read_csv('/kaggle/input/titanic/test.csv')

gender_submission=pd.read_csv('/kaggle/input/titanic/gender_submission.csv')     
train.head()
test.head()
print(train.shape, test.shape)
train.info()
test.info()
train.isnull().sum()
test.isnull().sum()
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
def bar_chart(feature):

    survived=train[train['Survived']==1][feature].value_counts()

    dead=train[train['Survived']==0][feature].value_counts()

    df=pd.DataFrame([survived,dead])

    df.index=['survived','dead']

    df.plot(kind='bar',stacked=True, figsize=(10,5))
bar_chart('Sex')
bar_chart('Pclass')
bar_chart('Embarked')
bar_chart('SibSp')
bar_chart('Parch')
train['Age'].fillna(train['Age'].mean(),inplace=True)

test['Age'].fillna(test['Age'].mean(),inplace=True)

test['Fare'].fillna(test['Fare'].mean(),inplace=True)

train['Embarked'].fillna(value='S',inplace=True)
train['family']=train['SibSp']+train['Parch']+1

test['family']=test['SibSp']+train['Parch']+1
train['Sex'] = train['Sex'].replace(['female','male'],[0,1])

train['Embarked'] = train['Embarked'].replace(['S','Q','C'],[1,2,3])
test['Sex'] = test['Sex'].replace(['female','male'],[0,1])

test['Embarked'] = test['Embarked'].replace(['S','Q','C'],[1,2,3])
train_clean=train.drop(columns=['PassengerId','Name','SibSp','Parch','Ticket','Cabin'])

test_clean=test.drop(columns=['PassengerId','Name','SibSp','Parch','Ticket','Cabin'])
X_train=train_clean.drop(columns=['Survived'])

y_train=train_clean[['Survived']]
from sklearn.preprocessing import StandardScaler

X_train_scale=StandardScaler().fit_transform(X_train)

pd.DataFrame(X_train_scale).head()


from sklearn.linear_model import LogisticRegression

LR=LogisticRegression().fit(X_train_scale, y_train)

y_pred=LR.predict(test_clean)

from sklearn.metrics import classification_report

#print(classification_report(y_pred, gender_submission['Survived']))

from sklearn.model_selection import cross_val_score

scores=cross_val_score(LogisticRegression(),X_train_scale,y_train,cv=5)

print(scores)

print(scores.mean())
from sklearn.model_selection import GridSearchCV

grid={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}

score=GridSearchCV(LogisticRegression(),grid).fit(X_train_scale, y_train)

print(score.best_params_)

print(score.best_score_)
data = {'PassengerId':gender_submission['PassengerId'],

        'Survived':y_pred}

result=pd.DataFrame(data)

result.to_csv('/kaggle/working/result_lr.csv', index=False)

output=pd.read_csv('/kaggle/working/result_lr.csv')
from sklearn.ensemble import RandomForestClassifier

RF=RandomForestClassifier().fit(X_train_scale, y_train)

y_pred_rf=RF.predict(test_clean)

from sklearn.metrics import classification_report

#print(classification_report(y_pred_rf, gender_submission['Survived']))

#print(y_pred_rf)

scores=cross_val_score(RandomForestClassifier(), X_train_scale, y_train, cv=5)

print(scores)

print(scores.mean())
data={'PassengerId': gender_submission['PassengerId'],'Survived':y_pred_rf}

result_rf=pd.DataFrame(data)

result_rf.to_csv('/kaggle/working/result_rf.csv', index=False)

result_rf1=pd.read_csv('/kaggle/working/result_rf.csv')
from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score

svc=SVC(kernel='linear', C=1)

scores=cross_val_score(svc, X_train_scale, y_train, cv=5)

print(scores)

print(scores.mean())
y_pred_svc=SVC(kernel='linear', C=1).fit(X_train_scale, y_train).predict(test_clean)

data={'PassengerId': gender_submission['PassengerId'],'Survived':y_pred_svc}

result_svc=pd.DataFrame(data)

result_svc.to_csv('/kaggle/working/result_svc.csv', index=False)

result_svc=pd.read_csv('/kaggle/working/result_svc.csv')
from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import cross_val_score

sgd=SGDClassifier()

scores=cross_val_score(sgd, X_train, y_train, cv=5)

print(scores)

print(scores.mean())



y_pred_sgd=SGDClassifier().fit(X_train, y_train).predict(test_clean)

data={'PassengerId': gender_submission['PassengerId'],'Survived':y_pred_sgd}

result_sgd=pd.DataFrame(data)

result_sgd.head()



result_sgd.to_csv('/kaggle/working/result_sgd.csv', index=False)

result_sgd=pd.read_csv('/kaggle/working/result_sgd.csv')

result_sgd.head()
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score

dtc=DecisionTreeClassifier()

scores=cross_val_score(dtc, X_train, y_train, cv=5)

print(scores)

print(scores.mean())



y_pred_dtc=DecisionTreeClassifier().fit(X_train, y_train).predict(test_clean)

data={'PassengerId': gender_submission['PassengerId'],'Survived':y_pred_dtc}

result_dtc=pd.DataFrame(data)

result_dtc.to_csv('/kaggle/working/result_dtc.csv', index=False)

result_dtc=pd.read_csv('/kaggle/working/result_dtc.csv')
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import cross_val_score

nb=GaussianNB()

scores=cross_val_score(nb, X_train, y_train, cv=5)

print(scores)

print(scores.mean())



y_pred_nb=GaussianNB().fit(X_train, y_train).predict(test_clean)

data={'PassengerId': gender_submission['PassengerId'],'Survived':y_pred_nb}

result_nb=pd.DataFrame(data)

result_nb.to_csv('/kaggle/working/result_nb.csv', index=False)

result_nb=pd.read_csv('/kaggle/working/result_nb.csv')
import keras

import tensorflow as tf
model_nn=keras.Sequential([

    keras.layers.Dense(6,activation=tf.nn.relu, input_shape=[6]),

    keras.layers.Dense(8,activation=tf.nn.relu),

    keras.layers.Dense(1,activation='softmax')

    ])
#optimizer=tf.keras.optimizers.RMSprop(0.001)

model_nn.compile(loss='binary_crossentropy', optimizer='Adam',metrics=['accuracy'])

model_nn.fit(X_train,y_train, epochs=5)
y_pred_nn=model_nn.predict(test_clean).astype(int)
gender_submission['Survived']= y_pred_nn

gender_submission.to_csv('/kaggle/working/result_neural.csv', index=False)

result_nn=pd.read_csv('/kaggle/working/result_neural.csv')
import time

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn.metrics import classification_report

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV



from sklearn.metrics import accuracy_score
train=pd.read_csv('/kaggle/input/titanic/train.csv')

test=pd.read_csv('/kaggle/input/titanic/test.csv')

gender_submission=pd.read_csv('/kaggle/input/titanic/gender_submission.csv')    
train['Title'] = train['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

train['FamilySize'] = train['SibSp'] + train['Parch'] + 1

train['FareBin'] = pd.qcut(train['Fare'], 4)

train['AgeBin'] = pd.qcut(train['Age'], 5)



test['Title'] = test['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

test['FamilySize'] = test['SibSp'] + test['Parch'] + 1

test['FareBin'] = pd.qcut(test['Fare'], 4)

test['AgeBin'] = pd.qcut(test['Age'], 5)
X_train=train.drop(columns=['Survived','Cabin','Fare','Age','PassengerId','Ticket','SibSp','Parch','Name'])

Y_train=train.Survived

X_test=test.drop(columns=['Cabin','Fare','Age','PassengerId','Ticket','SibSp','Parch','Name'])
num_feat=X_train.select_dtypes(include='number').columns.to_list()

cat_feat=X_train.select_dtypes(include='object').columns.to_list()
num_pipe=Pipeline([

    ('imputer', SimpleImputer(strategy='median')),

    ('scale', StandardScaler())

])



cat_pipe=Pipeline([

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('coder', OneHotEncoder(handle_unknown='ignore'))

])



ct=ColumnTransformer(remainder='drop',

    transformers=[

    ('numerical',num_pipe, num_feat),

    ('categorical',cat_pipe, cat_feat)

])



model_new=Pipeline([

    ('transformer', ct),

    ('predictor', RandomForestClassifier(n_jobs=1,random_state=0))

])



model_new.fit(X_train, Y_train);
print('Default model score: ', model_new.score(X_train, Y_train))
y_pred_train=model_new.predict(X_train)

print('In sample Score: ', accuracy_score(y_pred_train, Y_train))
def submission(test, model):

    y_pred=model.predict(test)

    data={'PassengerId': gender_submission['PassengerId'],'Survived':y_pred}

    result=pd.DataFrame(data)

    #date=pd.Timestamp.now().strftime(format='%d_%m_%Y_%H-%M_')

    result.to_csv(f'/kaggle/working/pipeline_result.csv', index=False)
submission(X_test,model_new)