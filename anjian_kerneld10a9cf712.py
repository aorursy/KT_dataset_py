# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import scipy as sp

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns',14)

column_name=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']

data=pd.read_csv('../input/housing.csv',delim_whitespace=True, names=column_name)

#Preview data

print(data.head())
print(data.info()) 

#View the data set, the type of dataset features, and missing conditions.
print(data.describe())

#Let us look at the overall description of the data set distribution.
import seaborn as sns



fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(12, 8))

i = 0

axs = axs.flatten()

for key,value in data.items():   

    sns.boxplot(y=key, data=data, ax=axs[i],sym='*')

    i += 1

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)

plt.show()
i=0

fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(12, 8))

axs = axs.flatten()

for k,v in data.items():

    sns.distplot(v, bins=10,ax=axs[i])

    i += 1

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)



plt.show()
#let's plot the pairwise correlation on data now.

print(data.corr(method='pearson'))

plt.figure(figsize=(12,8))

sns.heatmap(data.corr(),annot=True)

plt.show()
#Taking arguments

x=data[['INDUS','RM','TAX','PTRATIO','LSTAT']]

#Dependent variable

y=data['MEDV']
# Variance expansion factor to judge multicollinearity

# def vif(df,col_i):

#     from statsmodels.formula.api import ols



#     cols=list(df.columns)

#     cols.remove(col_i)

#     cols_noti=cols

#     formula=col_i+'~'+'+'.join(cols_noti)

#     r2=ols(formula,df).fit().rsquared

#     return 1./(1.-r2)



# for i in x.columns:

#     print(i,'\t',vif(df=x,col_i=i))
#

plt.figure(figsize=(12,5))

for i,col in enumerate(x.columns):

        plt.subplot(1,5,i+1)

        plt.plot(x[col],y,'o')

        plt.xlabel(col)

        plt.ylabel('MEDV')

        plt.plot(np.unique(x[col]), np.poly1d(np.polyfit(x[col], y, 1))(np.unique(x[col])))



plt.show()

plt.savefig('regression_plot.png')
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge,RidgeCV

from sklearn.linear_model import Lasso,LassoCV

from sklearn.svm import SVR

from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import BaggingRegressor

from sklearn.ensemble import AdaBoostRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error,r2_score
#Standardized data

scaler=StandardScaler()

X=scaler.fit_transform(x)

#Divide the data set into a training set and a test set.

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=123)

kf=KFold(n_splits=10)
#LinearRegression

lr_model=LinearRegression()



# Ridge

Ridge_model=Ridge()

alpha=np.logspace(-2,3,100,base=10)

rcv=RidgeCV(alphas=alpha,store_cv_values=True)

rcv.fit(X_train,y_train)

print('the best alpha of Ridge is {}'.format(rcv.alpha_)) #6.7341506577508214

Ridge_model.set_params(alpha=6.7)



#Lasso

Lasso_model=Lasso()

alpha_L=np.logspace(-2,3,100,base=10)

Lcv=LassoCV(alphas=alpha_L,cv=10)

Lcv.fit(X_train,y_train)

print('the best alpha of Lasso is {}'.format(rcv.alpha_)) #6.7341506577508214

Lasso_model.set_params(alpha=6.7)



#SVR

kernel=('linear','rbf')

gamma=np.arange(0.001,1.0,0.1)

C=np.arange(0.001,1.0,1)

grid={'kernel':kernel,'gamma':gamma,'C':C}

svr_search=GridSearchCV(estimator=SVR(),param_grid=grid,cv=10)

svr_search.fit(X_train,y_train)

print(svr_search.best_params_)  #{'C': 0.001, 'gamma': 0.001, 'kernel': 'linear'}

SVR_model=SVR(kernel='linear',gamma=0.001,C=0.001)



#DecisionTreeRegressor

Tree_model=DecisionTreeRegressor(max_depth=5)



#KNeighborsRegressor

KNN_model=KNeighborsRegressor(n_neighbors=5)#KNN_model:-24.950518 8.678390



# ensemble

BR_model=BaggingRegressor()

ABR_model=AdaBoostRegressor()

RFR_model=RandomForestRegressor()

GBR_model=GradientBoostingRegressor()



# neural_network

MLPR_model=MLPRegressor(hidden_layer_sizes=(100,),activation='logistic',alpha=0.001,max_iter=1000)
lr_score=cross_val_score(lr_model,X_train,y_train,cv=kf,scoring='neg_mean_squared_error')

Ridge_score=cross_val_score(Ridge_model,X_train,y_train,cv=kf,scoring='neg_mean_squared_error')

Lasso_score=cross_val_score(Lasso_model,X_train,y_train,cv=kf,scoring='neg_mean_squared_error')

SVR_score=cross_val_score(SVR_model,X_train,y_train,cv=kf,scoring='neg_mean_squared_error')

DTR_score=cross_val_score(Tree_model,X_train,y_train,cv=kf,scoring='neg_mean_squared_error')

KNN_score=cross_val_score(KNN_model,X_train,y_train,cv=kf,scoring='neg_mean_squared_error')

BR_score=cross_val_score(BR_model,X_train,y_train,cv=kf,scoring='neg_mean_squared_error')

ABR_score=cross_val_score(ABR_model,X_train,y_train,cv=kf,scoring='neg_mean_squared_error')

RFR_score=cross_val_score(RFR_model,X_train,y_train,cv=kf,scoring='neg_mean_squared_error')

GBR_score=cross_val_score(GBR_model,X_train,y_train,cv=kf,scoring='neg_mean_squared_error')

MLPR_score=cross_val_score(MLPR_model,X_train,y_train,cv=kf,scoring='neg_mean_squared_error')



print('lr_model:%f %f'% (lr_score.mean(), lr_score.std()))          

print('Ridge_model:%f %f'% (Ridge_score.mean(), Ridge_score.std())) 

print('Lasso_model:%f %f'%(Lasso_score.mean(),Lasso_score.std()))   

print('SVR_model:%f %f'%(SVR_score.mean(),SVR_score.std()))         

print('DTR_model:%f %f'%(DTR_score.mean(),DTR_score.std()))         

print('KNN_model:%f %f'%(KNN_score.mean(),KNN_score.std()))         

print('BR_model:%f %f'%(BR_score.mean(),BR_score.std()))            

print('ABR_model:%f %f'%(ABR_score.mean(),ABR_score.std()))         

print('RFR_model:%f %f'%(RFR_score.mean(),RFR_score.std()))         

print('GBR_model:%f %f'%(GBR_score.mean(),GBR_score.std()))         

print('MLPR_model:%f %f'%(MLPR_score.mean(),MLPR_score.std()))      
model=GradientBoostingRegressor()

kfold=KFold(n_splits=10)

param_Grid=dict(n_estimators=np.array([50,100,150,200,250,300,350,400,450,500]))

Grid=GridSearchCV(estimator=model,param_grid=param_Grid,scoring='neg_mean_squared_error',cv=kfold)

result=Grid.fit(X_train,y_train)

print(result.best_score_,result.best_params_)
model=GradientBoostingRegressor(n_estimators=150,random_state=123).fit(X_train,y_train)

prediction=model.predict(X_test)

print('mean_squared_error:',mean_squared_error(y_test, prediction))

print('r2_score:',r2_score(y_test, prediction))