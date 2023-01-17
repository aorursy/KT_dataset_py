

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dtrain=pd.read_csv('/kaggle/input/my-dataset/credit_train.csv')

dtest=pd.read_csv('/kaggle/input/my-dataset/credit_test.csv')
dtrain.head()
dtrain.shape, dtest.shape
# dropping the samples whose target (loan status) having missing values

dtrain=dtrain.drop(dtrain['Loan Status'][dtrain['Loan Status'].isna()].index)
dtrain['Loan Status'][dtrain['Loan Status'].isna()]
X=dtrain.iloc[:,2:]

dtest=dtest.iloc[:,2:]
X.info()
# credit score range is 300-850, above than 850s or less than 300s are outliers

X[X['Credit Score']>850].shape, dtest[dtest['Credit Score']>850].shape,
# removing outliers

X=X.drop(X[X['Credit Score']>850].index)

dtest=dtest.drop(dtest[dtest['Credit Score']>850].index)
from sklearn.impute import SimpleImputer
cat=[]

nom=[]

for i in X:

    if X[i].dtype=='object':

        cat.append(i)

    else:

        nom.append(i)

cat=cat[1:]

# remove target
# imputer for nominal (on mean value), for categorical (on most_frequent value)

nominal_imp=SimpleImputer(strategy='mean')

cat_imp=SimpleImputer(strategy='most_frequent')
def impute(df):

    for i in cat:

        df[i]=cat_imp.fit_transform(df[i][:,np.newaxis])

    for i in nom:

        df[i]=nominal_imp.fit_transform(df[i][:,np.newaxis])

    return df
X=impute(X)

dtest=impute(dtest)
X.head()
def target_based_barchart(df, feature):

    fully_paid=df[feature][df['Loan Status']=='Fully Paid'].value_counts()

    charged_off=df[feature][df['Loan Status']=='Charged Off'].value_counts()

    d=pd.DataFrame([fully_paid,charged_off])

    d.index=['Fully Paid','Charged Off']

    d.plot(kind='bar',stacked=True)
target_based_barchart(X,'Term')
target_based_barchart(X,'Home Ownership')
X.groupby(['Term','Loan Status'])['Annual Income'].mean()
X.info()
import matplotlib.pyplot as plt

import seaborn as sns
X.groupby(['Term','Loan Status'])['Loan Status'].count()
sns.catplot('Loan Status', col='Term', data=X, kind='count')
X['Years in current job'].unique()
X['Years in current job']=X['Years in current job'].map({'8 years':8, '10+ years':10, '3 years':3, '5 years':5, '< 1 year':0,

       '2 years':2, '4 years':4, '9 years':9, '7 years':7, '1 year':1, '6 years':6})



dtest['Years in current job']=dtest['Years in current job'].map({'8 years':8, '10+ years':10, '3 years':3, '5 years':5, '< 1 year':0,

       '2 years':2, '4 years':4, '9 years':9, '7 years':7, '1 year':1, '6 years':6})
dtest['Years in current job'].value_counts()
cat
cat.remove(cat[1])
def enc(df):

    for i in cat:

        df=pd.concat([df, pd.get_dummies(df[i], prefix=i, drop_first=True)], 1)

        df=df.drop(i,axis=1)

    return df
X=enc(X)

dtest=enc(dtest)
X.head(2)
from copy import deepcopy

X1=deepcopy(X)

dt1=deepcopy(dtest)
features=X1.iloc[:,1:]

target=X1['Loan Status']
from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.feature_selection import SelectKBest, chi2

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.model_selection import train_test_split, KFold, cross_val_score

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA



import warnings

warnings.filterwarnings("ignore")
x_train, x_val, y_train, y_val=train_test_split(features, target, shuffle=False)

x_test=dt1
scale=StandardScaler()

features=scale.fit_transform(features)

dt1=scale.fit_transform(dt1)
kf= KFold(n_splits=10, shuffle=False)
model=LogisticRegression()

score=cross_val_score(model, features, target, scoring='accuracy', cv=kf)

print(score)

print('avg score:', round(np.mean(score)*100),2)
model=RandomForestClassifier()

score=cross_val_score(model, features, target, scoring='accuracy', cv=kf)

print(score)

print('avg score:', round(np.mean(score)*100),2)
model=LogisticRegression()

model.fit(x_train, y_train)

ypred=model.predict(x_test)