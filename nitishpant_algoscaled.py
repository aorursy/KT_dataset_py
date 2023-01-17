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
import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

import warnings

from sklearn.decomposition import PCA

from sklearn.feature_selection import RFE

from sklearn.feature_selection import RFECV

from sklearn.feature_selection import SelectKBest, chi2

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, f1_score

from sklearn.metrics import f1_score,confusion_matrix

from sklearn.model_selection import train_test_split
import xgboost as xgb
xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,

                max_depth = 5, alpha = 10, n_estimators = 10)
df=pd.read_csv('/kaggle/input/assignment.csv')
df
pd.set_option('display.max_columns', None)
df.head()
print(df['Good_Bad'].value_counts())

print(df['LOCATIONNAME'].value_counts())

print(df['ADDRESSCONFIRMED'].value_counts())

print(df['VEHICLEMODEL'].value_counts())

print(df['LANDOWNERSHIP'].value_counts())

print(df['IRRIGATIONSOURCE'].value_counts())

print(df['CROPSCULTIVATED'].value_counts())

print(df['ASSETREGMONTH'].value_counts())
del df['LANDOWNERSHIP']

del df['IRRIGATIONSOURCE']

del df['CROPSCULTIVATED']

del df['ASSETREGMONTH']
df.info()
df.head()
df.drop(df[df.isnull().any(axis=1)].index,inplace=True)
for _ in df.columns:

    print("The number of null values in:{} == {}".format(_, df[_].isnull().sum()))
df['RESIDENCETYPE'].value_counts()
p={'Good':1,'Bad':0}  #Good_bad

q={'Y':1,'N':0} #isfam  #addconfirmed #politicalink #imgconfirm

r={'O':1,'R':2,'L':3}

s={'Y':1,'N':2,'0':0} #stabilityconfirmed

u={'O':1,'R':2,'L':3,'Y':4,'0':5,'N':6} #offtype

v={'SEP':1,'SAL':2,'AGR':3,'OTH':4,'PEN':5,'STU':6} #protype

df['Good_Bad']=df['Good_Bad'].map(p).astype('int64')
df['ADDRESSCONFIRMED']=df['ADDRESSCONFIRMED'].map(q).astype('int64')

df['ISFAMILYINVOLVED']=df['ISFAMILYINVOLVED'].map(q).astype('int64')

df['POLITICALLINK']=df['POLITICALLINK'].map(q).astype('int64')

df['IMGCONFIRM']=df['IMGCONFIRM'].map(q).astype('int64')

df['STABILITYCONFIRMED']=df['STABILITYCONFIRMED'].map(s).astype('int64')

df['OFFICETYPE']=df['OFFICETYPE'].map(u).astype('int64')

df['PROFESSIONTYPE']=df['PROFESSIONTYPE'].map(v).astype('int64')
df['RESIDENCETYPE']=df['RESIDENCETYPE'].map(r).astype('int64')
df.info()
df.head()
categorical_feature_columns = list(set(df.columns) - set(df._get_numeric_data().columns))

categorical_feature_columns
numerical_feature_columns = list(df._get_numeric_data().columns)

numerical_feature_columns
target='Good_Bad'
k = 15 #number of variables for heatmap

cols = df[numerical_feature_columns].corr().nlargest(k, target)[target].index

cm = df[cols].corr()

plt.figure(figsize=(10,6))

sns.heatmap(cm, annot=True, cmap = 'viridis')
X = df[numerical_feature_columns]

Y = df.loc[:, df.columns == target]
print(X.shape)

print(Y.shape)
x_train, x_test, y_train, y_test = train_test_split(X, Y,test_size=0.33,random_state=8)
clf_lr = LogisticRegression()      

lr_baseline_model = clf_lr.fit(x_train,y_train)
def generate_accuracy_and_heatmap(model, x, y):

#     cm = confusion_matrix(y,model.predict(x))

#     sns.heatmap(cm,annot=True,fmt="d")

    ac = accuracy_score(y,model.predict(x))

    f_score = f1_score(y,model.predict(x))

    print('Accuracy is: ', ac)

    print('F1 score is: ', f_score)

    print ("\n")

    print (pd.crosstab(pd.Series(model.predict(x), name='Predicted'),

                       pd.Series(y['Good_Bad'],name='Actual')))

    return 1
generate_accuracy_and_heatmap(lr_baseline_model, x_test, y_test)
rfe = RFE(estimator=clf_lr, step=1)

rfe = rfe.fit(x_train, y_train)
selected_rfe_features = pd.DataFrame({'Feature':list(x_train.columns),

                                      'Ranking':rfe.ranking_})

selected_rfe_features.sort_values(by='Ranking')
x_train_rfe = rfe.transform(x_train)

x_test_rfe = rfe.transform(x_test)
x_train_rfe[0:3]
lr_rfe_model = clf_lr.fit(x_train_rfe, y_train)
generate_accuracy_and_heatmap(lr_rfe_model, x_test_rfe, y_test)
rfecv = RFECV(estimator=clf_lr, step=1, cv=5, scoring='accuracy')

rfecv = rfecv.fit(x_train, y_train)

print('Optimal number of features :', rfecv.n_features_)

print('Best features :', x_train.columns[rfecv.support_])
rfecv.grid_scores_
plt.figure()

plt.xlabel("Number of features selected")

plt.ylabel("Cross validation score of number of selected features")

plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)

plt.show()
x_train_rfecv = rfecv.transform(x_train)

x_test_rfecv = rfecv.transform(x_test)
lr_rfecv_model = clf_lr.fit(x_train_rfecv, y_train)
generate_accuracy_and_heatmap(lr_rfecv_model, x_test_rfecv, y_test)