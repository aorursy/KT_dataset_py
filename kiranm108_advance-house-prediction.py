import pandas as pd

import numpy as np

import seaborn as sb

pd.pandas.set_option('display.max_columns',None)

import matplotlib.pyplot as plt

Train = pd.read_csv(r"../input/house-prices-data/train.csv",na_values=[np.nan,'NA','NaN'])

Test = pd.read_csv(r"../input/house-prices-data/test.csv",na_values=[np.nan,'NA','NaN'])
y_train = Train['SalePrice']
X = Train.copy()

X.head()
X = pd.concat([X,Train],axis=0)
Temporal_feature = [feat for feat in X.columns if 'Yr' in feat or 'Year' in feat]

Temporal_feature
Temporal_feature_nan = [feat for feat in Temporal_feature if X[feat].isnull().sum()>0]

Temporal_feature_nan
print(X['GarageYrBlt'].mode())

print(X['GarageYrBlt'].median())

print(X['GarageYrBlt'].mean())
X['GarageYrBlt'] = X['GarageYrBlt'].fillna(X['GarageYrBlt'].median())
X['GarageYrBlt'].isnull().sum()
numerical_feature = [feat for feat in X.columns if X[feat].dtypes!='O' and feat not in Temporal_feature]

len(numerical_feature)
numerical_feature_nan = [feat for feat in numerical_feature if X[feat].isnull().sum()>0]

numerical_feature_nan
X[numerical_feature_nan] = X[numerical_feature_nan].fillna(X[numerical_feature_nan].median())
X[numerical_feature_nan].isnull().sum()
categorical_feature = [feat for feat in X.columns if X[feat].dtypes=='O']

len(categorical_feature)
categorical_feature_nan = [feat for feat in categorical_feature if X[feat].isnull().sum()>0]

categorical_feature_nan
# X.drop('Id',axis=1,inplace=True)

X.shape
categorical_feature_nan_drop = [feat for feat in categorical_feature_nan if X[feat].isnull().sum()/len(X)>=0.7]

categorical_feature_nan_drop
X.drop(categorical_feature_nan_drop,axis=1,inplace=True)
categorical_feature = [feat for feat in X.columns if X[feat].dtypes=='O']

len(categorical_feature)
X['MasVnrType'] = X['MasVnrType'].fillna( X['MasVnrType'].value_counts().index[0])

categorical_feature_nan = [feat for feat in categorical_feature if X[feat].isnull().sum()>0]

categorical_feature_nan
for i in categorical_feature_nan:

    X[i].fillna(X[i].value_counts().index[0],inplace=True)
X[categorical_feature_nan].isnull().sum()
# X.drop('Id',axis=1,inplace=True)

X.head()
df = pd.DataFrame(pd.get_dummies(X[categorical_feature],drop_first=True))

df.head()
X.drop(categorical_feature,axis=1,inplace=True)
X.shape
X = pd.concat([X,df],axis=1)
X.head()
X.shape
Cormat = X.corr()

Set = set()

for i in range(len(Cormat.columns)):

    for j in range(i):

        if np.abs(Cormat.iloc[i,j])>0.85:

            col = Cormat.columns[i]

            Set.add(col)
sb.heatmap(Cormat)
Set
X.drop(Set,axis=1,inplace=True)
X.shape
Train['SaleType'].value_counts()
variance = [feat for feat in X.columns if np.var(X[feat])<0.01]

variance
X.drop(variance,axis=1,inplace=True)
X.shape
Train.shape[0]
X_train = X[:Train.shape[0]]
X_train.shape
X_test = X[Train.shape[0]:]
X_test.shape
from sklearn.feature_selection import f_classif
feature_fvalue = f_classif(X_train,y_train)[1]

feature_fvalue
feature = X_train.columns[feature_fvalue<0.05]

feature
X_train = X_train[feature]

X_test = X_test[feature]
print(X_train.shape,X_test.shape)
from sklearn.preprocessing import StandardScaler
X_train = pd.DataFrame(StandardScaler().fit_transform(X_train))

X_train.head()

X_test = pd.DataFrame(StandardScaler().fit_transform(X_test))

X_test.head()
from sklearn.linear_model import LinearRegression
lrg=LinearRegression()
lrg.fit(X_train,y_train)
LR = pd.Series(lrg.predict(X_test))
from sklearn.model_selection import cross_val_score

from sklearn.metrics import r2_score,make_scorer
print(cross_val_score(lrg,X_train,y_train,cv=5,scoring = make_scorer(r2_score)))
from sklearn.ensemble import RandomForestRegressor
print(np.mean(cross_val_score(RandomForestRegressor(n_estimators=200),X_train,y_train,cv=5,scoring = make_scorer(r2_score))))
from sklearn.svm import SVC
print(np.mean(cross_val_score(SVC(),X_train,y_train,cv=5,scoring = make_scorer(r2_score))))
from xgboost import XGBRegressor
print(np.mean(cross_val_score(XGBRegressor(learning_rate = 0.11),X_train,y_train,cv=5,scoring = make_scorer(r2_score))))
RRT = RandomForestRegressor(n_estimators=200)
RRT.fit(X_train,y_train)
RT =pd.Series(RRT.predict(X_test))
DATA = pd.concat([LR,RT],axis=1)
DATA.columns = ['LR','RT']

DATA.head()
DATA['diff'] = DATA.LR-DATA.RT
DATA.head()
sb.heatmap(DATA.corr(),annot=True)
#So by looking at it Linear regression give 1 r2 score and xgboost gives me 0.997 which is really good