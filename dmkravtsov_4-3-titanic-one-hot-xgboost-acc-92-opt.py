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
import seaborn as sns

import pandas_profiling

pd.set_option('display.max_rows', 1000)

import matplotlib.pyplot as plt

import math

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, roc_auc_score ,roc_curve,auc

from sklearn.model_selection import GridSearchCV,StratifiedKFold

from sklearn.preprocessing import LabelEncoder

from collections import Counter

seed =45



plt.style.use('fivethirtyeight')
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
## thanks to @Nadezda Demidova  https://www.kaggle.com/demidova/titanic-eda-tutorial-with-seaborn

train.loc[train['PassengerId'] == 631, 'Age'] = 48



# Passengers with wrong number of siblings and parch

train.loc[train['PassengerId'] == 69, ['SibSp', 'Parch']] = [0,0]

test.loc[test['PassengerId'] == 1106, ['SibSp', 'Parch']] = [0,0]
## checking for Survived dependence of Sex feature

train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_target = train.Survived

test_id = test.PassengerId
## let's concatenate test and train datasets excluding ID and Target features

df = pd.concat((train.loc[:,'Pclass':'Embarked'], test.loc[:,'Pclass':'Embarked'])).reset_index(drop=True)



report = pandas_profiling.ProfileReport(df)

display(report)
# df feature distribution before features tuning

def basic_details(df):

    b = pd.DataFrame()

    b['Missing value, %'] = round(df.isnull().sum()/df.shape[0]*100)

    b['N unique value'] = df.nunique()

    b['dtype'] = df.dtypes

    return b

basic_details(df)
# simplest NaN imputation



for col in df:

    if df[col].dtype == 'object':        

        df[col].fillna('N', inplace=True)

    else: df[col].fillna(df[col].median(), inplace=True)

        
    

for col in df:

    if df[col].nunique()<=2:

        df[col] = df[col].astype(str)

df = pd.get_dummies(df)
# df feature distribution after features tuning

def basic_details(df):

    b = pd.DataFrame()

    b['Missing value, %'] = round(df.isnull().sum()/df.shape[0]*100)

    b['N unique value'] = df.nunique()

    b['dtype'] = df.dtypes

    return b

basic_details(df)
train = df.iloc[:train.shape[0], :]

test = df.iloc[train.shape[0]:, :]





X = train

y = train_target
import xgboost as xgb

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from imblearn.over_sampling import RandomOverSampler

from imblearn.over_sampling import SMOTE, ADASYN

from imblearn.under_sampling import RandomUnderSampler



sm = SMOTE(random_state=42)

X_sm, y_sm = sm.fit_sample(X, y)



rus = RandomUnderSampler(random_state=42)

X_rus, y_rus = rus.fit_resample(X, y)



ros = RandomOverSampler(random_state=42)

X_ros, y_ros = ros.fit_resample(X, y)



adasyn = ADASYN(random_state=42)

X_ad, y_ad = adasyn.fit_resample(X, y)





x_train, x_valid, y_train, y_valid = train_test_split(X_ros, y_ros, test_size=0.2, random_state=10)



d_train = xgb.DMatrix(x_train, label=y_train)

d_valid = xgb.DMatrix(x_valid, label=y_valid)

d_test = xgb.DMatrix(test)



params = {

        'objective':'binary:hinge',

        'eta': 0.3,

        'max_depth':12,

        'learning_rate':0.03,

        'eval_metric':'auc',

        'min_child_weight':1,

        'subsample':1,

        'colsample_bytree':0.4,

        'seed':29,

        'reg_lambda':2.8,

        'reg_alpha':0,

        'gamma':0,

        'scale_pos_weight':1,

        'n_estimators': 600,

        'nthread':-1

}



watchlist = [(d_train, 'train'), (d_valid, 'valid')]

nrounds=10000  

model = xgb.train(params, d_train, nrounds, watchlist, early_stopping_rounds=600, 

                           maximize=True, verbose_eval=10)
fig,ax = plt.subplots(figsize=(15,20))

xgb.plot_importance(model,ax=ax,max_num_features=20,height=0.8,color='g')

#plt.tight_layout()

plt.show()
from sklearn import metrics

y_pred = model.predict(d_valid)



print('Accuracy :{0:0.5f}'.format(metrics.accuracy_score(y_valid, y_pred))) 

print('AUC : {0:0.5f}'.format(metrics.roc_auc_score(y_valid, y_pred)))

print('Precision : {0:0.5f}'.format(metrics.precision_score(y_valid, y_pred)))

print('Recall : {0:0.5f}'.format(metrics.recall_score(y_valid, y_pred)))

print('F1 : {0:0.5f}'.format(metrics.f1_score(y_valid, y_pred)))
# leaks = {

# 897:1,

# 899:1, 

# 930:1,

# 932:1,

# 949:1,

# 987:1,

# 995:1,

# 998:1,

# 999:1,

# 1016:1,

# 1047:1,

# 1083:1,

# 1097:1,

# 1099:1,

# 1103:1,

# 1115:1,

# 1118:1,

# 1135:1,

# 1143:1,

# 1152:1, 

# 1153:1,

# 1171:1,

# 1182:1,

# 1192:1,

# 1203:1,

# 1233:1,

# 1250:1,

# 1264:1,

# 1286:1,

# 935:0,

# 957:0,

# 972:0,

# 988:0,

# 1004:0,

# 1006:0,

# 1011:0,

# 1105:0,

# 1130:0,

# 1138:0,

# 1173:0,

# 1284:0,

# }
sub = pd.DataFrame()

sub['PassengerId'] = test_id

sub['Survived'] = model.predict(d_test)

sub['Survived'] = sub['Survived'].apply(lambda x: 1 if x>0.8 else 0)

# sub['Survived'] = sub.apply(lambda r: leaks[int(r['PassengerId'])] if int(r['PassengerId']) in leaks else r['Survived'], axis=1)

# sub.to_csv('submission.csv', index=False)



sub.head()