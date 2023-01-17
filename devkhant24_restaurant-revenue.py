# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import xgboost

from scipy import stats

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from datetime import date

import warnings



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/restaurant-revenue-prediction/train.csv')
df.drop(['Id'],axis=1,inplace=True)
df['Open Date'] = pd.to_datetime(df['Open Date'])

#df['Year'] = df['Open Date'].dt.year

#df['Month'] = df['Open Date'].dt.month

df['passed_years'] = date.today().year - df['Open Date'].dt.year 

df['passed_months'] = (date.today().year - df['Open Date'].dt.year) * 12 + date.today().month - df['Open Date'].dt.month
df.drop('Open Date',axis=1,inplace=True)
dft = pd.read_csv('/kaggle/input/restaurant-revenue-prediction/test.csv')
dft.drop(['Id'],axis=1,inplace=True)
dft['Open Date'] = pd.to_datetime(dft['Open Date'])

dft['passed_years'] = date.today().year - dft['Open Date'].dt.year

dft['passed_months'] = (date.today().year - dft['Open Date'].dt.year) * 12 + date.today().month - dft['Open Date'].dt.month 
dft.drop('Open Date',axis=1,inplace=True)
def onehot(mcolumns):

    dffinal = finaldf

    i=0

    for fields in mcolumns:

        print(fields)

        df1 = pd.get_dummies(finaldf[fields],drop_first=True)

        

        finaldf.drop([fields],axis=1,inplace=True)

        if i==0:

            dffinal=df1.copy()

        else:

            dffinal=pd.concat([dffinal,df1],axis=1)

        i=i+1

            

    dffinal=pd.concat([finaldf,dffinal],axis=1)

    

    return dffinal            
finaldf= pd.concat([df,dft],axis=0,sort=False)
finaldf = onehot(['Type','City','City Group'])
finaldf.shape
finaldf = finaldf.loc[:,~finaldf.columns.duplicated()]
dftrain = finaldf.iloc[:137,:]

dftest = finaldf.iloc[137:,:]
#z = np.abs(stats.zscore(b))

#threshold = 3

#b = b[(z < 3).all(axis=1)]
#z = np.abs(stats.zscore(c))

#threshold = 3

#c = c[(z < 3).all(axis=1)]
dftest.drop(['revenue'],axis=1,inplace=True)
#dftrain.columns.to_series().groupby(dftrain.dtypes).groups
c = dftest[['passed_months','P1', 'P10', 'P11', 'P12', 'P14', 'P15', 'P16', 'P17', 'P18', 'P19',

        'P20', 'P21', 'P22', 'P23', 'P24', 'P25', 'P30', 'P31', 'P32', 'P33',

        'P34', 'P35', 'P36', 'P37', 'P5', 'P6', 'P7', 'P8', 'P9',

         'passed_years','P13', 'P2', 'P26', 'P27', 'P28', 'P29', 'P3', 'P4']]
xtrain = dftrain[['passed_months','P1', 'P10', 'P11', 'P12', 'P14', 'P15', 'P16', 'P17', 'P18', 'P19',

        'P20', 'P21', 'P22', 'P23', 'P24', 'P25', 'P30', 'P31', 'P32', 'P33',

        'P34', 'P35', 'P36', 'P37', 'P5', 'P6', 'P7', 'P8', 'P9',

         'passed_years','P13', 'P2', 'P26', 'P27', 'P28', 'P29', 'P3', 'P4']]

ytrain = dftrain['revenue']
#,'P30','P32','P34','P36','P19','P31','P16','P23','P35','P18','P14','P26','P15','P17'
bestfeature = SelectKBest(score_func = chi2,k = 10)

fit = bestfeature.fit(xtrain,ytrain)
dfscore = pd.DataFrame(fit.scores_)

dfcolumn = pd.DataFrame(xtrain.columns)
featurescore = pd.concat([dfcolumn,dfscore],axis=1)

featurescore.columns=['Specs','Score']
print(featurescore.nsmallest(15,'Score'))
warnings.simplefilter(action='ignore',category=FutureWarning)
#score = cross_val_score(xg,xtrain,ytrain,cv=5)
#score.mean()
#Xtrain ,Xtest,Ytrain,Ytest = train_test_split(xtrain,ytrain,test_size=0.25,random_state=1)
booster = ['gbtree','gblinear']

base_score = [0.25,0.5,0.75,1]

n_estimators = [100,500,900,1100,1300,1500]

max_depth = [2,3,5,9,11,15]

learning_rate =[0.05,1,0.15,0.20]

min_child_weight = [1,2,3,4]



hyperparameter_grid = {

    'n_estimators':n_estimators,

    'max_depth':max_depth,

    'learning_rate':learning_rate,

    'min_child_weight':min_child_weight,

    'booster':booster,

    'base_score':base_score,

}
random_cv = RandomizedSearchCV(estimator=xg,

                              param_distributions=hyperparameter_grid,

                              cv=5,n_iter=50,

                              scoring='neg_mean_absolute_error',n_jobs=4,

                              verbose=5,

                              return_train_score=True,

                              random_state=1)
random_cv.fit(xtrain,ytrain)
random_cv.best_estimator_
xg = xgboost.XGBRegressor(base_score=0.75, booster='gbtree', colsample_bylevel=1,

             colsample_bynode=1, colsample_bytree=1, gamma=0,

             importance_type='gain', learning_rate=0.05, max_delta_step=0,

             max_depth=11, min_child_weight=1, missing=None, n_estimators=900,

             n_jobs=1, nthread=None, objective='reg:linear', random_state=0,

             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,

             silent=None, subsample=1, verbosity=1)

xg.fit(xtrain,ytrain)
xg.score(xtrain,ytrain)
result = xg.predict(c)
pred = pd.DataFrame(result)

pred.columns=['revenue']
test = pd.concat([pred,dftest],axis=1)
train = pd.concat([dftrain,test],axis=0,sort=False)
X = train[['passed_months','P1', 'P10', 'P11', 'P12', 'P14', 'P15', 'P16', 'P17', 'P18', 'P19',

        'P20', 'P21', 'P22', 'P23', 'P24', 'P25', 'P30', 'P31', 'P32', 'P33',

        'P34', 'P35', 'P36', 'P37', 'P5', 'P6', 'P7', 'P8', 'P9',

         'passed_years','P13', 'P2', 'P26', 'P27', 'P28', 'P29', 'P3', 'P4']]

Y = train['revenue']
booster = ['gbtree','gblinear']

base_score = [0.25,0.5,0.75,1]

n_estimators = [100,500,900,1100,1300,1500]

max_depth = [2,3,5,9,11,15]

learning_rate =[0.05,1,0.15,0.20]

min_child_weight = [1,2,3,4]



hyperparameter_grid = {

    'n_estimators':n_estimators,

    'max_depth':max_depth,

    'learning_rate':learning_rate,

    'min_child_weight':min_child_weight,

    'booster':booster,

    'base_score':base_score,

}
random_cv = RandomizedSearchCV(estimator=xg,

                              param_distributions=hyperparameter_grid,

                              cv=5,n_iter=50,

                              scoring='neg_mean_absolute_error',n_jobs=4,

                              verbose=5,

                              return_train_score=True,

                              random_state=1)
xg = xgboost.XGBRegressor()

xg.fit(X,Y)
ypred = xg.predict(c)
#xg.score(X,Y)
Pred = pd.DataFrame(ypred)

sub = pd.read_csv('/kaggle/input/restaurant-revenue-prediction/sampleSubmission.csv')

dataset = pd.concat([sub['Id'],Pred],axis=1)

dataset.columns=['Id','Prediction']

dataset.to_csv('sampleSubmission.csv',index=False)