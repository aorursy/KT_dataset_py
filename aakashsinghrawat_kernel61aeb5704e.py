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
train=pd.read_csv('/kaggle/input/airline-passenger-satisfaction/train.csv')

test=pd.read_csv('/kaggle/input/airline-passenger-satisfaction/test.csv')
train.head()
train=train.drop(train.columns[0:2],1)

test=test.drop(test.columns[0:2],1)
train.shape, test.shape
train.describe()
def get_cat(df):

    """Gets list of categorical features from passed dataframe"""

    cat=[]

    for i in df.columns:

        if df[i].dtypes=='object':

            cat.append(i) 

    return cat
cat1=get_cat(train)

cat2=get_cat(test)
train[train.isna().any(1)].shape
test[test.isna().any(1)].shape
def get_nom(df):

    """Gets nominal feature"""

    nom=[]

    for i in df.columns:

        if df[i].dtypes!='object':

            nom.append(i)

    return nom[2:]  # no need of feature id and age
nom1=get_nom(train)

nom2=get_nom(test)
from sklearn.impute import SimpleImputer
imp=SimpleImputer(strategy='median')

# impute with median
def impute(df, nom):

    "Impute for nan values"

    for i in nom:

        df[i]=imp.fit_transform(df[i][:, np.newaxis])

    return df
train=impute(train, nom1)

test=impute(test, nom2)
train[train.isna().any(1)].shape, test[test.isna().any(1)].shape

# no missing values
import matplotlib.pyplot as plt
def bar_plot(df,col):

    """Plot bar graph based on passed categorical features"""

    

    satisfied=df[col][df['satisfaction']=='satisfied'].value_counts()

    neutral=df[col][df['satisfaction']=='neutral or dissatisfied'].value_counts()

    dt=pd.DataFrame([satisfied,neutral], index=['satisfied','neutral or dissatisfied'])

    dt.plot.bar(stacked=True)
bar_plot(train,'Class')
bar_plot(train, 'Gender')
bar_plot(train, 'Type of Travel')
cat1=cat1[0:4]

cat2=cat2[0:4]
def cat_enc(df,cat):

    """Encoding (by get_dummies) for categorical features.

    Dropping the original feature before getting new dummy for each category in feature

    """

    dummies=[]

    for i in cat:

        dummies=pd.get_dummies(df[i], drop_first=True, prefix=i)

        df=df.drop(i,axis=1)

        df=pd.concat([df,dummies],1)

    return df
train=cat_enc(train, cat1)

test=cat_enc(test, cat2)
train.head(1)
# encoding for our target feature (satisfaction)

train['satisfaction']=train['satisfaction'].map({'satisfied':1,'neutral or dissatisfied':0})

test['satisfaction']=test['satisfaction'].map({'satisfied':1,'neutral or dissatisfied':0})

# no need to target feature in test dataframe since we are about to predict it

#test.drop('satisfaction',1,inplace=True)
dtrain=train.copy()

dtest=test.copy()
train.head(1)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

import warnings

warnings.filterwarnings('ignore')
X=dtrain.drop('satisfaction',1)

y=dtrain['satisfaction']

x_test=dtest.drop('satisfaction',1)
scale=StandardScaler()

X=scale.fit_transform(X)

x_test=scale.fit_transform(x_test)
x_train, x_val, y_train, y_val=train_test_split(X,y,shuffle=False)
skf=StratifiedKFold(n_splits=10, shuffle=False)
logi_reg=LogisticRegression()

score= cross_val_score(logi_reg, X, y, scoring='accuracy', cv=skf)

print(score)

print('avg score:',np.mean(score)*100)
rfc=RandomForestClassifier(5)

score= cross_val_score(rfc, X, y, scoring='accuracy', cv=skf)

print(score)

print('avg score:',np.mean(score)*100)
gbc=GradientBoostingClassifier(n_estimators=20)

score= cross_val_score(gbc, X, y, scoring='accuracy', cv=skf)

print(score)

print('avg score:',np.mean(score)*100)
## rfc looks good here to me

## lets predict using this

rfc=RandomForestClassifier(5)

rfc.fit(x_train, y_train)

ypred=rfc.predict(x_test)
from sklearn.metrics import accuracy_score
accuracy_score(dtest['satisfaction'], ypred)