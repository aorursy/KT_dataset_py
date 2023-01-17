#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""

Created on Mon Jul  8 15:36:20 2019



@author: mkp

"""



import pandas as pd



from sklearn.preprocessing import MinMaxScaler,StandardScaler

from sklearn import svm

from sklearn.svm import SVC,LinearSVC

from sklearn.linear_model import SGDClassifier

from sklearn.linear_model import SGDRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, precision_score, recall_score

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np



from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats



from sklearn.model_selection import KFold

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import cross_val_score

from sklearn.metrics import r2_score



from sklearn.decomposition import PCA

import pandas_profiling as pdp



import warnings

warnings.filterwarnings('ignore')

#%matplotlib inline

########################################################################

def build_data(df):

    

    for column in df.select_dtypes(include=object):

        labels, uniques = pd.factorize(df[column])

        df[column] = labels

        mm = MinMaxScaler()

        df[column] = mm.fit_transform(df[column].values.reshape(-1, 1))



    for column in df.select_dtypes(include=int):

        df[column] = df[column].fillna(df[column].median())

        ss = MinMaxScaler()

        df[column] = ss.fit_transform(df[column].values.reshape(-1, 1))

        #print(df[column].head())

        

    for column in df.select_dtypes(include=float):

        df[column] = df[column].fillna(df[column].median())

        ss = MinMaxScaler()

        df[column] = ss.fit_transform(df[column].values.reshape(-1, 1))

        #print(df[column].head())



    if 'SalePrice' in df.columns.values:

        df = df.drop(columns=['SalePrice'], axis=1) 



    return df



def corr(df):

    df = build_data(df)

    df_corr = df.corr()

    print(df_corr)

    print(type(df_corr))

    df_corr.to_csv('df_corr.csv',index=False)   



def pca_make(df):  

    

    pca = PCA(n_components=.99)

    df = pca.fit_transform(df)

    pca_df_ratio = pca.explained_variance_ratio_

    print(pca_df_ratio)



    return df



def profile_report_make(df):  

    

    profile_report = pdp.ProfileReport(df_master)

    profile_report.to_file('report.html')

    from IPython.display import HTML

    HTML(filename='report.html')



    return df

########################################################################



df = pd.read_csv('../input/train.csv', dtype = None, delimiter = ",")

df_test = pd.read_csv('../input/test.csv', dtype = None, delimiter = ",")



df_Id = df_test['Id']



df.head()



df_master = pd.concat([df, df_test],sort=False)



df_master = df_master.drop(columns=['Exterior1st','BsmtExposure','GarageArea','GarageType','GarageCond','GarageQual','GarageFinish','MasVnrType','BsmtCond','BsmtFinType1','BsmtFinType2','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','BsmtFullBath','BsmtHalfBath'], axis=1) 

#profile_report_make(df_master)



########################################################################



df_master = build_data(df_master)





print(df_master.isnull().any())

print(df_master.describe())

print(df_master.dtypes)



plt.figure(figsize=(20, 20))

sns.heatmap(df_master.corr(), cmap='BuPu', annot=False)

plt.show()



########################################################################



X = df_master[0:1460]



df_test = df_master[1460:]



y = df['SalePrice']



########################################################################



pca = PCA(n_components=.99)

X = pca.fit_transform(X)

pca_df_ratio = pca.explained_variance_ratio_

print(pca_df_ratio)



df_test = pca.transform(df_test)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0,shuffle=False)



########################################################################



clf = SVC(kernel='linear', random_state=30, gamma='scale',probability=True)

clf.fit(X_train, y_train)



predict_data = clf.predict(X_test)

print("accuracy_score :", accuracy_score(y_test, predict_data))



########################################################################

model = LinearRegression()  #モデルの呼び出し

model.fit(X_train, y_train)  



print(model.coef_)  

print(model.intercept_)  

print(model.get_params())

print('########################################################################')

print(model.score(X_test,y_test))  



kf = KFold(n_splits=3, shuffle=False, random_state=0)

scores = cross_val_score(clf, X_train, y_train, cv=kf)

print(scores.mean())



predict_data = model.predict(df_test)

predict_data = pd.DataFrame({'SalePrice':predict_data})

predict_data = pd.concat([df_Id, predict_data], axis=1, sort=False)

predict_data.to_csv('Submission1.csv',index=False)

########################################################################

model = LogisticRegression()  #モデルの呼び出し

model.fit(X_train, y_train)  



print(model.coef_)  

print(model.intercept_)  

print(model.get_params())

print('########################################################################')

print({'score':model.score(X_test,y_test)})  



kf = KFold(n_splits=3, shuffle=False, random_state=0)

scores = cross_val_score(clf, X_train, y_train, cv=kf)

print(scores.mean())



predict_data = model.predict(df_test)

predict_data = pd.DataFrame({'SalePrice':predict_data})

predict_data = pd.concat([df_Id, predict_data], axis=1, sort=False)

predict_data.to_csv('Submission2.csv',index=False)

########################################################################

########################################################################

clf = GradientBoostingRegressor(random_state=0)

clf.fit(X_train, y_train)



y_pred = clf.predict(X_test)

acc = clf.score(X_test, y_test)

print('ACC: %.4f' % acc)

print({'r2_score':r2_score(y_test,y_pred)})



kf = KFold(n_splits=3, shuffle=False, random_state=0)

scores = cross_val_score(clf, X_train, y_train, cv=kf)

print(scores.mean())



predict_data = clf.predict(df_test)

predict_data = pd.DataFrame({'SalePrice':predict_data})

predict_data = pd.concat([df_Id, predict_data], axis=1, sort=False)

predict_data.to_csv('Submission3.csv',index=False)

########################################################################