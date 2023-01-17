import pandas as pd

import numpy as np

import sklearn as skl

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
data_train = pd.read_csv('../input/train_data.csv')

data_test = pd.read_csv('../input/test_data.csv')
data_train.columns
data_train=data_train.drop(['Unnamed: 0'],axis=1).drop(['Unnamed: 1'],axis=1)

data_test =data_test.drop(['Unnamed: 0'],axis=1).drop(['Unnamed: 1'],axis=1)
corr_matrix=data_train.corr()

f,ax=plt.subplots(figsize=(30,20))

sns.heatmap(corr_matrix,vmax=1,annot=True)
for i in list(data_train.columns):

    if data_train[i].nunique()==1:

        print(i)

        print('Drop Column:',i)

        data_train=data_train.drop([i],axis=1)

        data_test =data_test.drop([i],axis=1)
highly_correlated_feature     = list(corr_matrix.index[abs(corr_matrix['SalePrice'])>0.7])

moderately_correlated_feature = list(corr_matrix.index[abs(corr_matrix['SalePrice'])>0.4])

modestely_correlated_feature  = list(corr_matrix.index[abs(corr_matrix['SalePrice'])>0.1])

weakly_correlated_feature     = list(corr_matrix.index[abs(corr_matrix['SalePrice'])>0.01])



highly_correlated_feature.remove('SalePrice')

moderately_correlated_feature.remove('SalePrice')

modestely_correlated_feature.remove('SalePrice')

weakly_correlated_feature.remove('SalePrice')



print("highly_correlated_feature:\n",highly_correlated_feature,'\n')

print("moderately_correlated_feature:\n",moderately_correlated_feature,'\n')

print("modestely_correlated_feature:\n",len(modestely_correlated_feature),'columns\n')

print("weakly_correlated_feature:\n",len(weakly_correlated_feature),'columns')
useless_features=list(corr_matrix.index[abs(corr_matrix["SalePrice"])<0.01])

print("UselessFeatures:",useless_features)
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV
sns.pairplot(data_train, x_vars = highly_correlated_feature, y_vars='SalePrice', size=7, aspect=0.8, kind='reg')
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()

fig.set_size_inches(12.5, 7.5)

ax = fig.add_subplot(111, projection='3d')

ax.scatter(xs=data_train['GrLivArea'], ys=data_train['OverallQual'], zs=data_train['SalePrice'])

ax.set_ylabel('OverallQual'); ax.set_xlabel('GrLivArea'); ax.set_zlabel('SalePrice')

ax.view_init(20, -45)
from sklearn.linear_model import LinearRegression

LR_model = LinearRegression()

LR_parameters={}

LR_GSCV= GridSearchCV(LR_model,LR_parameters,refit=True,cv=3)



list_correlated_feature = [highly_correlated_feature,moderately_correlated_feature,modestely_correlated_feature,weakly_correlated_feature]



for correlated_feature in list_correlated_feature:

    X_train, X_test, y_train, y_test = train_test_split(data_train[correlated_feature], data_train['SalePrice'],test_size=0.25, random_state=33)

    LR_GSCV.fit(X_train,y_train)

    LR_GSCV_result = LR_GSCV.predict(data_test[correlated_feature])

    params_LR = LR_GSCV.best_params_

    best_score_LR=LR_GSCV.best_score_

    print('Features:',len(correlated_feature),'Columns')

    print('best_params_:',params_LR)

    print('best_score_:',best_score_LR)

    print('GSCV_result:\n',LR_GSCV_result)

    submission = pd.DataFrame({'Id':data_test['Id'],'SalePrice':LR_GSCV_result})

    filename = 'House_Price_LinearRegression_'+str(len(correlated_feature))+'features.csv'

    submission.to_csv(filename,index=False)

    print('Saved file: ' + filename)

    print('\n','-'*77,'\n')
LR_model = LinearRegression()

LR_parameters={}

LR_GSCV= GridSearchCV(LR_model,LR_parameters,refit=True,cv=3)



list_correlated_feature = [highly_correlated_feature,moderately_correlated_feature,modestely_correlated_feature,weakly_correlated_feature]

for i in range(2,3):

    print('degree = ',i,'\n\n')

    for correlated_feature in list_correlated_feature:

        X_train, X_test, y_train, y_test = train_test_split(data_train[correlated_feature], data_train['SalePrice'],test_size=0.25, random_state=33)

        from sklearn.preprocessing import PolynomialFeatures

        PolynomialFeatures = PolynomialFeatures(degree = i)

        X_train = PolynomialFeatures.fit_transform(X_train)

        LR_GSCV.fit(X_train,y_train)

        LR_GSCV_result = LR_GSCV.predict(PolynomialFeatures.fit_transform(data_test[correlated_feature]))

        params_LR = LR_GSCV.best_params_

        best_score_LR=LR_GSCV.best_score_

        print('Features:',len(correlated_feature),'Columns')

        print('best_params_:',params_LR)

        print('best_score_:',best_score_LR)

        print('GSCV_result:\n',LR_GSCV_result)

        submission = pd.DataFrame({'Id':data_test['Id'],'SalePrice':LR_GSCV_result})

        filename = 'House_Price_LinearRegression_degree'+ str(i)+'_'+str(len(correlated_feature))+'features.csv'

        submission.to_csv(filename,index=False)

        print('Saved file: ' + filename,'\n\n')

        print('\n','-'*77,'\n')
from sklearn.tree import DecisionTreeRegressor

DTR_model = DecisionTreeRegressor()

DTR_parameters={'random_state':list(range(10))}

DTR_GSCV= GridSearchCV(DTR_model,DTR_parameters,refit=True,cv=3)



list_correlated_feature = [highly_correlated_feature,moderately_correlated_feature,modestely_correlated_feature,weakly_correlated_feature]



for correlated_feature in list_correlated_feature:

    X_train, X_test, y_train, y_test = train_test_split(data_train[correlated_feature], data_train['SalePrice'],test_size=0.25, random_state=33)

    DTR_GSCV.fit(X_train,y_train)

    DTR_GSCV_result = DTR_GSCV.predict(data_test[correlated_feature])

    params_DTR = DTR_GSCV.best_params_

    best_score_DTR=DTR_GSCV.best_score_

    print('Features:',len(correlated_feature),'Columns')

    print('best_params_:',params_DTR)

    print('best_score_:',best_score_DTR)

    print('GSCV_result:\n',DTR_GSCV_result)

    submission = pd.DataFrame({'Id':data_test['Id'],'SalePrice':DTR_GSCV_result})

    filename = 'House_Price_DecisionTreeRegressor_'+str(len(correlated_feature))+'features.csv'

    submission.to_csv(filename,index=False)

    print('Saved file: ' + filename)

    print('\n','-'*77,'\n')
from sklearn.ensemble import RandomForestRegressor

RFR_model = RandomForestRegressor()

RFR_parameters={'random_state':list(range(10)),'n_estimators':list(range(50,100,2)),'max_depth':list(range(3,10))}

RFR_GSCV= GridSearchCV(RFR_model,RFR_parameters,refit=True,cv=3)



list_correlated_feature = [highly_correlated_feature,moderately_correlated_feature,modestely_correlated_feature,weakly_correlated_feature]



for correlated_feature in list_correlated_feature:

    X_train, X_test, y_train, y_test = train_test_split(data_train[correlated_feature], data_train['SalePrice'],test_size=0.25, random_state=33)

    RFR_GSCV.fit(X_train,y_train)

    RFR_GSCV_result = RFR_GSCV.predict(data_test[correlated_feature])

    params_RFR = RFR_GSCV.best_params_

    best_score_RFR=RFR_GSCV.best_score_

    print('Features:\n',len(correlated_feature),'Columns')

    print('best_params_:',params_RFR)

    print('best_score_:',best_score_RFR)

    print('GSCV_result:\n',RFR_GSCV_result)

    submission = pd.DataFrame({'Id':data_test['Id'],'SalePrice':RFR_GSCV_result})

    filename = 'House_Price_RandomForestRegressor_'+str(len(correlated_feature))+'features.csv'

    submission.to_csv(filename,index=False)

    print('Saved file: ' + filename)

    print('\n','-'*77,'\n')
from xgboost import XGBRegressor



XGBR_model = XGBRegressor()

XGBR_parameters={'random_state':list(range(10)),'max_depth':list(range(3,10)),'n_estimators':list(range(50,100,2))}

XGBR_GSCV= GridSearchCV(XGBR_model,XGBR_parameters,refit=True,cv=3)



list_correlated_feature = [highly_correlated_feature,moderately_correlated_feature,modestely_correlated_feature,weakly_correlated_feature]



for correlated_feature in list_correlated_feature:

    X_train, X_test, y_train, y_test = train_test_split(data_train[correlated_feature], data_train['SalePrice'],test_size=0.25, random_state=33)

    XGBR_GSCV.fit(X_train,y_train)

    XGBR_GSCV_result = XGBR_GSCV.predict(data_test[correlated_feature])

    params_XGBR = XGBR_GSCV.best_params_

    best_score_XGBR=XGBR_GSCV.best_score_

    print('Features:',len(correlated_feature),'Columns')

    print('best_params_:',params_XGBR)

    print('best_score_:',best_score_XGBR)

    print('GSCV_result:\n',XGBR_GSCV_result)

    submission = pd.DataFrame({'Id':data_test['Id'],'SalePrice':XGBR_GSCV_result})

    filename = 'House_Price_XGBRegressor_'+str(len(correlated_feature))+'features.csv'

    submission.to_csv(filename,index=False)

    print('Saved file: ' + filename)

    print('\n','-'*77,'\n')