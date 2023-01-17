# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#Visualisation Library

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



train.shape, test.shape
train=train.iloc[:,1:302]

train.head()
sns.set_style('white')

sns.set_color_codes(palette='deep')

f,ax= plt.subplots(figsize = (8,7))

sns.distplot(train['target'])
features=train.iloc[:,1:302]

features.shape

label=train.iloc[:,0]

features.head()

label.head()

plt.figure(figsize=(26, 24))

for i, col in enumerate(list(features.columns)[1:28]):

    plt.subplot(7, 4, i + 1)

    plt.hist(features[col])

    plt.title(col)
plt.figure(figsize=(26, 24))

for i, col in enumerate(list(features.columns)[28:56]):

    plt.subplot(7, 4, i + 1)

    plt.hist(features[col])

    plt.title(col)
plt.figure(figsize=(26, 24))

for i, col in enumerate(list(features.columns)[56:84]):

    plt.subplot(7, 4, i + 1)

    plt.hist(features[col])

    plt.title(col)
plt.figure(figsize=(26, 24))

for i, col in enumerate(list(features.columns)[112:140]):

    plt.subplot(7, 4, i + 1)

    plt.hist(features[col])

    plt.title(col)
plt.figure(figsize=(26, 24))

for i, col in enumerate(list(features.columns)[140:168]):

    plt.subplot(7, 4, i + 1)

    plt.hist(features[col])

    plt.title(col)
plt.figure(figsize=(26, 24))

for i, col in enumerate(list(features.columns)[168:199]):

    plt.subplot(7, 5, i + 1)

    plt.hist(features[col])

    plt.title(col)
a=[]

for f in features:

    a.append(features.iloc[:,int(f)].isnull().sum())

print(a)


# for j in range(0,300):

#     Q3 = features.iloc[:,j].quantile(0.75)

#     Q1 = features.iloc[:,j].quantile(0.25) 

#     iqr=Q3-Q1

#     upperBound=Q3 + 1.5*iqr

#     lowerBound=Q1 - 1.5*iqr

#     for i in range(0,250) :

#         if (features.iloc[i,j] > upperBound ):

#             print(features.iloc[i,j],upperBound)

#             features[i,j]=features.iloc[:,j].mean()



#         if (features.iloc[i,j] < lowerBound):

#             print(features.iloc[i,j],lowerBound)

#             features[i,j]=features.iloc[:,j].mean()





  
corr = train.corr()

c=corr.iloc[:,0].sort_values(ascending=False)

c.iloc[120:135]
from sklearn.model_selection import StratifiedKFold,RepeatedStratifiedKFold

from sklearn.preprocessing import StandardScaler

X_train = features.drop(['75','195'],axis=1)

X=features.drop(['75','195'],axis=1)



y_train = label

X_test = test.drop(['id','75','195'], axis=1)



scaler = StandardScaler()

SX_train = scaler.fit_transform(X_train)

SX_test = scaler.transform(X_test)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

lr = LogisticRegression(class_weight='balanced', penalty='l1', C=0.1, solver='liblinear')

model_lr=lr.fit(SX_train,y_train)

scores=cross_val_score(lr, SX_train, y_train,cv=3)

print("Accuracy: %0.3f (std %0.3f)" % (scores.mean(), scores.std() * 2))

import xgboost as xgb

from sklearn.model_selection import RandomizedSearchCV

from xgboost import XGBClassifier

scaler = StandardScaler()

SX_train = scaler.fit_transform(X_train)

SX_test = scaler.transform(X_test)

dtrain = xgb.DMatrix(SX_train, label=y_train)

folds = StratifiedKFold(n_splits=20, shuffle=True, random_state=42)

params = {'learning_rate':[0.1,0.01,0.02,0.03,0.2,0.3],

        'min_child_weight': [1, 5, 10],

        'gamma': [0.5, 1, 1.5, 2, 5],

        'subsample': [0.6, 0.8, 1.0],

        'colsample_bytree': [0.6, 0.8, 1.0],

        'max_depth': [1,2,3, 4, 5]

        }

xgb = XGBClassifier(n_estimators=600, objective='binary:logistic')

random_search = RandomizedSearchCV(xgb, param_distributions=params,n_iter=5,scoring='roc_auc', n_jobs=4, 

                                   cv=folds.split(SX_train,y_train), 

                                   random_state=1001)

random_search.fit(SX_train, y_train)

print('Best score: {}'.format(random_search.best_score_))

print('Best parameters: {}'.format(random_search.best_params_))
import xgboost as xgb

from sklearn.model_selection import RandomizedSearchCV

from xgboost import XGBClassifier

for x in X_train.columns:

    print(x)

print(y_train)

scaler = StandardScaler()

XSX_train = scaler.fit_transform(X_train)

XSX_test = scaler.transform(X_test)

dtrain = xgb.DMatrix(SX_train, label=y_train)

folds = StratifiedKFold(n_splits=20, shuffle=True, random_state=42)

params = {'learning_rate':[0.1,0.01,0.02,0.03,0.2,0.3],

        'min_child_weight': [1, 5, 10],

        'gamma': [0.5, 1, 1.5, 2, 5],

        'subsample': [0.6, 0.8, 1.0],

        'colsample_bytree': [0.6, 0.8, 1.0],

        'max_depth': [1,2,3, 4, 5]

        }

xgb = XGBClassifier(learning_rate=0.1, n_estimators=400, max_depth=2,

                        min_child_weight=10, gamma=1, subsample=0.8, colsample_bytree=0.6,

                        objective='binary:logistic',  seed=27)

model_xgb=xgb.fit(XSX_train,y_train)
fea_imp = pd.DataFrame({'imp': model_xgb.feature_importances_})

print(fea_imp.sort_values(by='imp',ascending=True))
import eli5

eli5.show_weights(model_xgb)
top_features = [i[1:] for i in eli5.formatters.as_dataframe.explain_weights_df(model_xgb).feature if 'BIAS' not in i]

print(top_features)

XX_train = train[top_features]

X_test=test.drop(['id'], axis=1)

XX_test = X_test[top_features]

scaler = StandardScaler()

XSX_train = scaler.fit_transform(XX_train)

XSX_test = scaler.transform(XX_test)

xgb = XGBClassifier(learning_rate=0.1, n_estimators=400, max_depth=2,

                        min_child_weight=10, gamma=1, subsample=0.8, colsample_bytree=0.6,

                        objective='binary:logistic',  seed=27)

model_xgb=xgb.fit(XSX_train,y_train)

scores=cross_val_score(model_xgb, XSX_train, y_train,cv=5)

print("Accuracy: %0.3f (std %0.3f)" % (scores.mean(), scores.std() * 2))
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import StratifiedKFold



LogisticRegression(solver='liblinear', max_iter=10000)

folds = StratifiedKFold(n_splits=20, shuffle=True, random_state=42)

parameter_grid = {'class_weight' : ['balanced', None],

                  'penalty' : ['l2', 'l1'],

                  'C' : [0.001, 0.01, 0.08, 0.1, 0.15, 1.0, 10.0, 100.0],

                 }



grid_search = GridSearchCV(lr, param_grid=parameter_grid, cv=folds, scoring='roc_auc')

grid_search.fit(XSX_train, y_train)

print('Best score: {}'.format(grid_search.best_score_))

print('Best parameters: {}'.format(grid_search.best_params_))
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

lr = LogisticRegression(class_weight=None, penalty='l1', C=0.08, solver='liblinear')

model_lr=lr.fit(XSX_train,y_train)

scores=cross_val_score(lr, XSX_train, y_train,cv=5)

print("Accuracy: %0.3f (std %0.3f)" % (scores.mean(), scores.std() * 2))
pred=model_lr.predict(XSX_test)

sub=pd.read_csv("../input/sample_submission.csv")

sub["target"]=pred

sub.to_csv("output",index=False)