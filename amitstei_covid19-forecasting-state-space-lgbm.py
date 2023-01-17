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
train = pd.read_csv ('/kaggle/input/covid19-global-forecasting-week-4/train.csv')

test = pd.read_csv ('/kaggle/input/covid19-global-forecasting-week-4/test.csv')

train['CRPS']=train.Country_Region+train.Province_State.fillna('')

test['CRPS']=test.Country_Region+test.Province_State.fillna('')

train['serd']=train.groupby('CRPS').cumcount()

train.Date=pd.to_datetime(train.Date)

test.Date=pd.to_datetime(test.Date)

traintest=pd.concat((train[train.Date<min(test.Date)],test)).copy(deep=True)

traintest.ConfirmedCases[traintest.ConfirmedCases.isnull()]=1

traintest.loc[traintest.ConfirmedCases==0,'days_since_confirmed']=0

traintest.loc[traintest.ConfirmedCases>0,'days_since_confirmed']=traintest[traintest.ConfirmedCases>0].groupby('CRPS').cumcount() #The first is 0 to avoid leakakge

test=traintest[traintest.Date>=min(test.Date)].copy(deep=True)

test
train.Date=pd.to_datetime(train.Date)

train.set_index('Date',inplace=True,drop=False)

train['LConfirmedCases']=np.log1p(train['ConfirmedCases'])

train['LFatalities']=np.log1p(train['Fatalities'])

train['LDConfirmedCases']=train.groupby('CRPS')[['LConfirmedCases']].diff()

train['LDFatalities']=train.groupby('CRPS')[['LFatalities']].diff()

'''

train['serd']=train.groupby('CRPS').cumcount()



from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.metrics import mean_squared_error,mean_squared_log_error

arimas={}



from skopt import gbrt_minimize,gp_minimize,forest_minimize

from joblib import dump,load

import time 



errs=[]

x0=load('/kaggle/input/farimas-cov/farimas_cov.joblib')



for crps in train.CRPS.unique():

    def opt_arima(p):

        try:

            start_time = time.time()



            train_series=train.Fatalities[(train.CRPS==crps) & (train.serd<=70)]

            test_series=train.Fatalities[(train.CRPS==crps) & (train.serd>70)]

            mod=SARIMAX(train_series, exog=train.ConfirmedCases[(train.CRPS==crps) & (train.serd<=70)],test_series=train.LFatalities[(train.CRPS==crps) & (train.serd>70)], order=(p[0],p[1], p[2]), freq='D', dates=train_series.index.values,simple_differencing=False, enforce_stationarity=False, enforce_invertibility=False, 

                        hamilton_representation=False, concentrate_scale=False, trend_offset=1, use_exact_diffuse=True,seasonal_order=(p[3],p[4],p[5],7))

            res=mod.fit(maxiter=10, method='powell')

            forecast=res.forecast(len(test_series),exog=train.ConfirmedCases[(train.CRPS==crps) & (train.serd>70)])

            forecast[forecast<0]=0

            forecast=np.nan_to_num(forecast)

            crps_error=np.sqrt(mean_squared_log_error(test_series[10:],forecast[10:]))

            print(p,crps_error)

            elapsed_time = time.time() - start_time

        except:

            crps_error=1000000

        return crps_error

    print(crps)

    ret=forest_minimize(opt_arima,[[0,5],[0,2],[0,5],[0,2],[0,2],[0,2]],n_calls=20,n_random_starts=5)

    print(ret.fun,ret.x)

    errs.append(ret.fun)

    arimas[crps]=ret.x

    dump(arimas,'farimas_cov.joblib')

'''
np.median(errs)
'''

train['serd']=train.groupby('CRPS').cumcount()



from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.metrics import mean_squared_error,mean_squared_log_error

arimas={}



from skopt import gbrt_minimize,gp_minimize

from joblib import dump,load

import time 



errs=[]

x0=load('/kaggle/input/ccarimas/ccarimas.joblib')



for crps in train.CRPS.unique():

    def opt_arima(p):

        try:

            train_series=train.ConfirmedCases[(train.CRPS==crps) & (train.serd<=70)]

            test_series=train.ConfirmedCases[(train.CRPS==crps) & (train.serd>70)]

            mod=SARIMAX(train_series, order=(p[0],p[1], p[2]), freq='D', dates=train_series.index.values,simple_differencing=False, enforce_stationarity=False, enforce_invertibility=False, 

                        hamilton_representation=False, concentrate_scale=False, trend_offset=1, use_exact_diffuse=True,seasonal_order=(p[3],p[4],p[5],7))

            res=mod.fit(maxiter=10, method='powell')

            forecast=res.forecast(len(test_series))

            forecast[forecast<0]=0

            forecast=np.nan_to_num(forecast)

            crps_error=np.sqrt(mean_squared_log_error(test_series,forecast))

            print(p,crps_error)

        except:

            crps_error=1000000

        return crps_error

    print(crps)

    ret=gbrt_minimize(opt_arima,[[0,10],[0,2],[0,10],[0,2],[0,2],[0,2]],n_calls=20,n_random_starts=5,x0=x0[crps])

    print(ret.fun,ret.x)

    errs.append(ret.fun)

    arimas[crps]=ret.x

    dump(arimas,'ccarimas.joblib')

'''
train
print(np.mean(errs))

np.median(errs)
from joblib import load

arimas=load('/kaggle/input/farimas-cov/farimas_cov.joblib')

pd.DataFrame(arimas).transpose().iloc[:,0].value_counts()

#2

#1

#3..5

#1

#1

#1

#7/13


from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.metrics import mean_squared_log_error

from joblib import load

arimas=load('/kaggle/input/final-arimas/ccarimas (1).joblib')

errors=[]

train.Date=pd.to_datetime(train.Date)

test.Date=pd.to_datetime(test.Date)

for crps in train.CRPS.unique():

        print(crps)

        train_series=train.ConfirmedCases[(train.CRPS==crps) & (train.serd<=70)]

        test_series=train.ConfirmedCases[(train.CRPS==crps) & (train.serd>70)]

        p=arimas[crps]

        mod=SARIMAX(train_series, order=(p[0],p[1], p[2]), freq='D', dates=train_series.index.values,simple_differencing=False, enforce_stationarity=False, enforce_invertibility=False, 

                    hamilton_representation=False, concentrate_scale=False, trend_offset=1, use_exact_diffuse=True,seasonal_order=(p[3],p[4],p[5],7))

        res=mod.fit(maxiter=200, method='powell')

        forecast=res.forecast(len(test[test.CRPS==crps]))

        forecast[forecast<0]=0

        forecast=np.nan_to_num(forecast)

        crps_error=np.sqrt(mean_squared_log_error(test_series,forecast[:len(test_series)]))

        print(crps_error)

        errors.append(crps_error)

        train.loc[(train.CRPS==crps) & (train.Date>=min(test.Date)),'ARIMApred']=forecast[:len(test_series)]

        test.loc[(test.CRPS==crps),'ARIMApred']=forecast

       

print(np.mean(errors)) #0.65

print(np.median(errors)) #0.21

%%time

arimas=load('/kaggle/input/finals/farimas_cov (2).joblib')

errors=[]

for crps in train.CRPS.unique():

        print(crps)

        train_series=train.Fatalities[(train.CRPS==crps) & (train.serd<=70)]

        test_series=train.Fatalities[(train.CRPS==crps) & (train.serd>70)]

        p=arimas[crps]

        mod=SARIMAX(train_series, exog=train.ConfirmedCases[(train.CRPS==crps) & (train.serd<=70)],order=(p[0],p[1], p[2]), freq='D', dates=train_series.index.values,simple_differencing=False, enforce_stationarity=False, enforce_invertibility=False, 

                    hamilton_representation=False, concentrate_scale=False, trend_offset=1, use_exact_diffuse=True,seasonal_order=(p[3],p[4],p[5],7))

        res=mod.fit(maxiter=200, method='powell')

        forecast=res.forecast(len(test[test.CRPS==crps]),exog=test.ARIMApred[test.CRPS==crps].values)

        forecast[forecast<0]=0

        forecast=np.nan_to_num(forecast)

        crps_error=np.sqrt(mean_squared_log_error(test_series,forecast[:len(test_series)]))

        print(crps_error)

        errors.append(crps_error)

        

        train.loc[(train.CRPS==crps) & (train.Date>=min(test.Date)),'FARIMApred']=forecast[:len(test_series)]

        test.loc[(test.CRPS==crps),'FARIMApred']=forecast

       

print(np.mean(errors)) #0.65

print(np.median(errors)) #0.21

test.loc[test.Country_Region=='Israel',['Date','ARIMApred','FARIMApred']]
train['LARIMApred']=np.log1p(train['ARIMApred'])

train['LDARIMApred']=train.groupby('CRPS')[['LARIMApred']].diff()

train['LFARIMApred']=np.log1p(train['FARIMApred'])

train['LFDARIMApred']=train.groupby('CRPS')[['LFARIMApred']].diff()

train['LConfirmedCases']=np.log1p(train['ConfirmedCases'])

train['LFatalities']=np.log1p(train['Fatalities'])

train['LDConfirmedCases']=train.groupby('CRPS')[['LConfirmedCases']].diff()

train['LDFatalities']=train.groupby('CRPS')[['LFatalities']].diff()



for i in range(1,15,1):

    train['LFatalities'+str(i)]=train.groupby('CRPS')[['LFatalities']].shift(i)

    train['LDFatalities'+str(i)]=train.groupby('CRPS')[['LDFatalities']].shift(i)

    train['LConfirmedCases'+str(i)]=train.groupby('CRPS')[['LConfirmedCases']].shift(i)

    train['LDConfirmedCases'+str(i)]=train.groupby('CRPS')[['LDConfirmedCases']].shift(i)



train['LDConfirmedCasesMA']=(train.LDConfirmedCases+train.LDConfirmedCases1+train.LDConfirmedCases2+train.LDConfirmedCases3+train.LDConfirmedCases4)/5

train['LDFatalitiesMA']=(train.LDFatalities+train.LDFatalities1+train.LDFatalities2+train.LDFatalities3+train.LDFatalities4)/5



for i in range(1,15,1):

    train['LDFatalitiesMA'+str(i)]=train.groupby('CRPS')[['LDFatalitiesMA']].shift(i)

    train['LDConfirmedCasesMA'+str(i)]=train.groupby('CRPS')[['LDConfirmedCasesMA']].shift(i)





train['serd']=train.groupby('CRPS').cumcount()

train.loc[train.ConfirmedCases==0,'days_since_confirmed']=0

train.loc[train.ConfirmedCases>0,'days_since_confirmed']=train[train.ConfirmedCases>0].groupby('CRPS').cumcount() #The first is 0 to avoid leakakge

from lightgbm import LGBMRegressor



lgbm_cc=LGBMRegressor(num_leaves = 76,learning_rate =10**-1.94,n_estimators=100,min_sum_hessian_in_leaf=(10**-4.2),min_child_samples =138,

                   colsample_bytree = 0.38,reg_lambda=10**1.35,random_state=1234,n_jobs=4)

lgbm_f=LGBMRegressor(num_leaves = 27,learning_rate =10**-1.79,n_estimators=100,min_sum_hessian_in_leaf=(10**-4.69),min_child_samples =54,subsample =0.8,subsample_freq=2,

                   colsample_bytree = 0.76,reg_lambda=10**1.08,random_state=1234,n_jobs=4)



from sklearn.preprocessing import OrdinalEncoder

oe=OrdinalEncoder()

X=oe.fit_transform(train[['Country_Region','Province_State']].fillna(''))

train['CR']=X[:,0]

train['PS']=X[:,1]



lgbm_cc.fit(train.loc[train.serd<=70,['LDConfirmedCases1','LDConfirmedCases2','LDConfirmedCases3','LDConfirmedCases4','LDConfirmedCases5','LDConfirmedCases6','LDConfirmedCases7','LDConfirmedCases8',

                                      'LDFatalities1','LDFatalities2','LDFatalities3','LDFatalities4','LDFatalities5','LDFatalities6','LDFatalities7','LDFatalities8',

                                         'CR','PS',

                                     'LDConfirmedCasesMA1','LDConfirmedCasesMA2','LDConfirmedCasesMA3','LDConfirmedCasesMA4','LDConfirmedCasesMA5','LDConfirmedCasesMA6','LDConfirmedCasesMA7','LDConfirmedCasesMA8',

                                     'LDFatalitiesMA1','LDFatalitiesMA2','LDFatalitiesMA3','LDFatalitiesMA4','LDFatalitiesMA5','LDFatalitiesMA6','LDFatalitiesMA7','LDFatalitiesMA8']],train.LDConfirmedCases[train.serd<=70],categorical_feature=['CR','PS'])

lgbm_f.fit(train.loc[train.serd<=70,['LDConfirmedCases1','LDConfirmedCases2','LDConfirmedCases3','LDConfirmedCases4','LDConfirmedCases5','LDConfirmedCases6','LDConfirmedCases7','LDConfirmedCases8',

                                      'LDFatalities1','LDFatalities2','LDFatalities3','LDFatalities4','LDFatalities5','LDFatalities6','LDFatalities7','LDFatalities8',

                                         'CR','PS',

                                     'LDConfirmedCasesMA1','LDConfirmedCasesMA2','LDConfirmedCasesMA3','LDConfirmedCasesMA4','LDConfirmedCasesMA5','LDConfirmedCasesMA6','LDConfirmedCasesMA7','LDConfirmedCasesMA8',

                                     'LDFatalitiesMA1','LDFatalitiesMA2','LDFatalitiesMA3','LDFatalitiesMA4','LDFatalitiesMA5','LDFatalitiesMA6','LDFatalitiesMA7','LDFatalitiesMA8']],train.LDFatalities[train.serd<=70],categorical_feature=['CR','PS'])

imp=pd.DataFrame()

imp['feature']=['LDConfirmedCases1','LDConfirmedCases2','LDConfirmedCases3','LDConfirmedCases4','LDConfirmedCases5','LDConfirmedCases6','LDConfirmedCases7','LDConfirmedCases8',

                                      'LDFatalities1','LDFatalities2','LDFatalities3','LDFatalities4','LDFatalities5','LDFatalities6','LDFatalities7','LDFatalities8',

                                         'CR','PS',

                                     'LDConfirmedCasesMA1','LDConfirmedCasesMA2','LDConfirmedCasesMA3','LDConfirmedCasesMA4','LDConfirmedCasesMA5','LDConfirmedCasesMA6','LDConfirmedCasesMA7','LDConfirmedCasesMA8',

                                     'LDFatalitiesMA1','LDFatalitiesMA2','LDFatalitiesMA3','LDFatalitiesMA4','LDFatalitiesMA5','LDFatalitiesMA6','LDFatalitiesMA7','LDFatalitiesMA8']

imp['imp']=lgbm_f.feature_importances_

pd.options.display.max_rows=999

imp
from sklearn.metrics import mean_squared_log_error

train['serd']=train.groupby('CRPS').cumcount()

trainpred = pd.concat((train[train.serd<=70],test)).reset_index(drop=True)





trainpred.sort_values(['Country_Region','Province_State','Date'],inplace=True)



X=oe.transform(trainpred[['Country_Region','Province_State']].fillna(''))

trainpred['CR']=X[:,0]

trainpred['PS']=X[:,1]



trainpred['serd']=trainpred.groupby('CRPS').cumcount()

trainpred.loc[trainpred.ConfirmedCases.isnull(),'ConfirmedCases']=1 #Heuristic

trainpred.loc[trainpred.ConfirmedCases==0,'days_since_confirmed']=0

trainpred.loc[trainpred.ConfirmedCases>0,'days_since_confirmed']=trainpred[trainpred.ConfirmedCases>0].groupby('CRPS').cumcount() #The first is 0 to avoid leakakge

trainpred['LConfirmedCases']=np.log1p(trainpred['ConfirmedCases'])

trainpred['LFatalities']=np.log1p(trainpred['Fatalities'])

trainpred['LDConfirmedCases']=trainpred.groupby('CRPS')[['LConfirmedCases']].diff()

trainpred['LDFatalities']=trainpred.groupby('CRPS')[['LFatalities']].diff()

trainpred['LARIMApred']=np.log1p(trainpred['ARIMApred'])

trainpred['LFARIMApred']=np.log1p(trainpred['FARIMApred'])

trainpred['LDARIMApred']=trainpred.groupby('CRPS')[['LARIMApred']].diff()

trainpred['LDFARIMApred']=trainpred.groupby('CRPS')[['LFARIMApred']].diff()



for serd in range(71,max(trainpred.serd)+1):

    print(serd)

    for i in range(1,8,1):

        trainpred['LFatalities'+str(i)]=trainpred.groupby('CRPS')[['LFatalities']].shift(i)

        trainpred['LDFatalities'+str(i)]=trainpred.groupby('CRPS')[['LDFatalities']].shift(i)

        trainpred['LConfirmedCases'+str(i)]=trainpred.groupby('CRPS')[['LConfirmedCases']].shift(i)

        trainpred['LDConfirmedCases'+str(i)]=trainpred.groupby('CRPS')[['LDConfirmedCases']].shift(i)

    trainpred['LDConfirmedCasesMA']=(trainpred.LDConfirmedCases+trainpred.LDConfirmedCases1+trainpred.LDConfirmedCases2+trainpred.LDConfirmedCases3+trainpred.LDConfirmedCases4)/5

    trainpred['LDFatalitiesMA']=(trainpred.LDFatalities+trainpred.LDFatalities1+trainpred.LDFatalities2+trainpred.LDFatalities3+trainpred.LDFatalities4)/5



    for i in range(1,8,1):

        trainpred['LDFatalitiesMA'+str(i)]=trainpred.groupby('CRPS')[['LDFatalitiesMA']].shift(i)

        trainpred['LDConfirmedCasesMA'+str(i)]=trainpred.groupby('CRPS')[['LDConfirmedCasesMA']].shift(i)



    trainpred.loc[trainpred.serd==serd,'LDConfirmedCases']= lgbm_cc.predict(trainpred.loc[trainpred.serd==serd,['LDConfirmedCases1','LDConfirmedCases2','LDConfirmedCases3','LDConfirmedCases4','LDConfirmedCases5','LDConfirmedCases6','LDConfirmedCases7','LDConfirmedCases8',

                                      'LDFatalities1','LDFatalities2','LDFatalities3','LDFatalities4','LDFatalities5','LDFatalities6','LDFatalities7','LDFatalities8',

                                         'CR','PS',

                                     'LDConfirmedCasesMA1','LDConfirmedCasesMA2','LDConfirmedCasesMA3','LDConfirmedCasesMA4','LDConfirmedCasesMA5','LDConfirmedCasesMA6','LDConfirmedCasesMA7','LDConfirmedCasesMA8',

                                     'LDFatalitiesMA1','LDFatalitiesMA2','LDFatalitiesMA3','LDFatalitiesMA4','LDFatalitiesMA5','LDFatalitiesMA6','LDFatalitiesMA7','LDFatalitiesMA8']])

    trainpred.loc[(trainpred.serd==serd) & (trainpred.LDConfirmedCases<0),'LDConfirmedCases']=0

    trainpred.loc[trainpred.serd==serd,'LConfirmedCases']=trainpred.loc[trainpred.serd==serd,'LDConfirmedCases']+trainpred.loc[trainpred.serd==serd,'LConfirmedCases1']

    trainpred.loc[trainpred.serd==serd,'ConfirmedCases']=np.exp(trainpred.loc[trainpred.serd==serd,'LConfirmedCases'])-1



    trainpred.loc[trainpred.serd==serd,'LDFatalities']= lgbm_f.predict(trainpred.loc[trainpred.serd==serd,['LDConfirmedCases1','LDConfirmedCases2','LDConfirmedCases3','LDConfirmedCases4','LDConfirmedCases5','LDConfirmedCases6','LDConfirmedCases7','LDConfirmedCases8',

                                      'LDFatalities1','LDFatalities2','LDFatalities3','LDFatalities4','LDFatalities5','LDFatalities6','LDFatalities7','LDFatalities8',

                                         'CR','PS',

                                     'LDConfirmedCasesMA1','LDConfirmedCasesMA2','LDConfirmedCasesMA3','LDConfirmedCasesMA4','LDConfirmedCasesMA5','LDConfirmedCasesMA6','LDConfirmedCasesMA7','LDConfirmedCasesMA8',

                                     'LDFatalitiesMA1','LDFatalitiesMA2','LDFatalitiesMA3','LDFatalitiesMA4','LDFatalitiesMA5','LDFatalitiesMA6','LDFatalitiesMA7','LDFatalitiesMA8']])

    trainpred.loc[(trainpred.serd==serd) & (trainpred.LDFatalities<0),'LDFatalities']=0

    trainpred.loc[trainpred.serd==serd,'LFatalities']=trainpred.loc[trainpred.serd==serd,'LDFatalities']+trainpred.loc[trainpred.serd==serd,'LFatalities1']

    trainpred.loc[trainpred.serd==serd,'Fatalities']=np.exp(trainpred.loc[trainpred.serd==serd,'LFatalities'])-1

    

print(np.sqrt(mean_squared_log_error (train.ConfirmedCases[train.serd>70],trainpred.ConfirmedCases[(trainpred.serd>70) & (trainpred.serd<=max(train.serd))])))

print(np.sqrt(mean_squared_log_error (train.Fatalities[train.serd>70],trainpred.Fatalities[(trainpred.serd>70) & (trainpred.serd<=max(train.serd))])))
from sklearn.model_selection import cross_val_predict,cross_val_score

from sklearn.linear_model import HuberRegressor,LinearRegression,BayesianRidge,RidgeCV,RANSACRegressor,ElasticNetCV

from sklearn.svm import SVR

from sklearn.neighbors import KNeighborsRegressor,NeighborhoodComponentsAnalysis

from sklearn.preprocessing import StandardScaler,RobustScaler,QuantileTransformer,PolynomialFeatures

from sklearn.ensemble import BaggingRegressor

from sklearn.metrics import mean_squared_error,mean_squared_log_error

import sklearn

qtx=StandardScaler()

qty=StandardScaler()

df=trainpred.loc[(trainpred.serd>70) & (trainpred.serd<=max(train.serd)),['LConfirmedCases','LARIMApred']]



print(np.sqrt(-np.mean(cross_val_score(SVR(kernel='rbf',C=10),qtx.fit_transform(df),qty.fit_transform(train.LConfirmedCases[train.serd>70].values.reshape(-1,1)),cv=10,scoring='neg_mean_squared_error'))))

print(np.sqrt(mean_squared_error(trainpred.LARIMApred[(trainpred.serd>70) & (trainpred.serd<=max(train.serd))],train.LConfirmedCases[train.serd>70])))

rc=SVR(kernel='rbf',C=10)

rc.fit(qtx.fit_transform(df),qty.transform(train.LConfirmedCases[train.serd>70].values.reshape(-1,1)))

trainpred.loc[(trainpred.serd>70) & (trainpred.serd<=max(train.serd)),'RLConfirmedCases']=qty.inverse_transform(cross_val_predict(SVR(kernel='rbf',C=10),qtx.transform(df),qty.transform(train.LConfirmedCases[train.serd>70].values.reshape(-1,1)),cv=10).reshape(-1,1))

print(np.sqrt(mean_squared_error(trainpred.RLConfirmedCases[(trainpred.serd>70) & (trainpred.serd<=max(train.serd))],train.LConfirmedCases[train.serd>70])))

#RobustScaler?

from sklearn.model_selection import cross_val_predict,cross_val_score

from sklearn.linear_model import HuberRegressor,LinearRegression,BayesianRidge,RidgeCV,RANSACRegressor,ElasticNetCV

from sklearn.svm import SVR

from sklearn.neighbors import KNeighborsRegressor,NeighborhoodComponentsAnalysis

from sklearn.preprocessing import StandardScaler,RobustScaler,QuantileTransformer,PolynomialFeatures

from sklearn.ensemble import BaggingRegressor

from sklearn.metrics import mean_squared_error,mean_squared_log_error

import sklearn

qtx_f=StandardScaler()

qty_f=StandardScaler()

df=trainpred.loc[(trainpred.serd>70) & (trainpred.serd<=max(train.serd)),['LConfirmedCases','LARIMApred','LFatalities','LFARIMApred','LDConfirmedCases','LDFatalities']]



print(np.sqrt(-np.mean(cross_val_score(SVR(kernel='rbf',C=1),qtx_f.fit_transform(df),qty_f.fit_transform(train.LFatalities[train.serd>70].values.reshape(-1,1)),cv=10,scoring='neg_mean_squared_error'))))

print(np.sqrt(mean_squared_error(trainpred.LFatalities[(trainpred.serd>70) & (trainpred.serd<=max(train.serd))],train.LFatalities[train.serd>70])))

rc_f=SVR(kernel='rbf',C=1)

rc_f.fit(qtx_f.fit_transform(df),qty_f.transform(train.LFatalities[train.serd>70].values.reshape(-1,1)))

trainpred.loc[(trainpred.serd>70) & (trainpred.serd<=max(train.serd)),'RLFatalities']=qty_f.inverse_transform(cross_val_predict(SVR(kernel='rbf',C=1),qtx_f.transform(df),qty_f.transform(train.LFatalities[train.serd>70].values.reshape(-1,1)),cv=10).reshape(-1,1))

print(np.sqrt(mean_squared_error(trainpred.RLFatalities[(trainpred.serd>70) & (trainpred.serd<=max(train.serd))],train.LFatalities[train.serd>70])))

for serd in range(71,max(trainpred.serd)+1):

    print(serd)

    

    if serd<=max(train.serd):

        trainpred.loc[trainpred.serd==serd,'LConfirmedCases']= trainpred.loc[trainpred.serd==serd,'RLConfirmedCases']

    else:

        trainpred.loc[trainpred.serd==serd,'LConfirmedCases']= qty.inverse_transform(rc.predict(qtx.transform(trainpred.loc[trainpred.serd==serd,['LConfirmedCases','LARIMApred']])).reshape(-1,1))

    trainpred.loc[trainpred.LConfirmedCases<0,'LConfirmedCases']=0

    trainpred.loc[trainpred.serd==serd,'LDConfirmedCases']=trainpred.loc[trainpred.serd==serd,'LConfirmedCases']-trainpred.loc[trainpred.serd==serd,'LConfirmedCases1']

    trainpred.loc[trainpred.LDConfirmedCases<0,'LDConfirmedCases']=0

    trainpred.loc[trainpred.serd==serd,'ConfirmedCases']=np.exp(trainpred.loc[trainpred.serd==serd,'LConfirmedCases'])-1    



    if serd<=max(train.serd):

        trainpred.loc[trainpred.serd==serd,'LFatalities']= trainpred.loc[trainpred.serd==serd,'RLFatalities']

    else:

        trainpred.loc[trainpred.serd==serd,'LFatalities']= qty_f.inverse_transform(rc_f.predict(qtx_f.transform(trainpred.loc[trainpred.serd==serd,['LConfirmedCases','LARIMApred','LFatalities','LFARIMApred','LDConfirmedCases','LDFatalities']])).reshape(-1,1))

    trainpred.loc[trainpred.LFatalities<0,'LFatalities']=0

    trainpred.loc[trainpred.serd==serd,'Fatalities']=np.exp(trainpred.loc[trainpred.serd==serd,'LFatalities'])-1    

    

public_trainpred=trainpred.copy()

print(np.sqrt(mean_squared_log_error (train.ConfirmedCases[train.serd>70],trainpred.ConfirmedCases[(trainpred.serd>70) & (trainpred.serd<=max(train.serd))])))

print(np.sqrt(mean_squared_log_error (train.Fatalities[train.serd>70],trainpred.Fatalities[(trainpred.serd>70) & (trainpred.serd<=max(train.serd))])))
public_trainpred.loc[public_trainpred.Country_Region.str.contains('Israel'),['Date','ConfirmedCases','Fatalities']].round()
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.metrics import mean_squared_log_error

from joblib import load

arimas=load('/kaggle/input/final-arimas/ccarimas (1).joblib')

for crps in train.CRPS.unique():

        print(crps)

        train_series=train.ConfirmedCases[(train.CRPS==crps)]

        p=arimas[crps]

        mod=SARIMAX(train_series, order=(p[0],p[1], p[2]), freq='D', dates=train_series.index.values,simple_differencing=False, enforce_stationarity=False, enforce_invertibility=False, 

                    hamilton_representation=False, concentrate_scale=False, trend_offset=1, use_exact_diffuse=True,seasonal_order=(p[3],p[4],p[5],7))

        res=mod.fit(maxiter=200, method='powell')

        forecast=res.forecast(len(test[test.CRPS==crps]))

        forecast[forecast<0]=0

        forecast=np.nan_to_num(forecast)

        test.loc[(test.CRPS==crps),'ARIMApred']=forecast

arimas=load('/kaggle/input/finals/farimas_cov (2).joblib')

for crps in train.CRPS.unique():

        print(crps)

        train_series=train.Fatalities[(train.CRPS==crps)]

        p=arimas[crps]

        mod=SARIMAX(train_series, exog=train.ConfirmedCases[(train.CRPS==crps)],order=(p[0],p[1], p[2]), freq='D', dates=train_series.index.values,simple_differencing=False, enforce_stationarity=False, enforce_invertibility=False, 

                    hamilton_representation=False, concentrate_scale=False, trend_offset=1, use_exact_diffuse=True,seasonal_order=(p[3],p[4],p[5],7))

        res=mod.fit(maxiter=200, method='powell')

        forecast=res.forecast(len(test[test.CRPS==crps]),exog=test.ConfirmedCases[(test.CRPS==crps)].values)

        forecast[forecast<0]=0

        forecast=np.nan_to_num(forecast)

        test.loc[(test.CRPS==crps),'FARIMApred']=forecast
from lightgbm import LGBMRegressor

lgbm_cc=LGBMRegressor(num_leaves = 76,learning_rate =10**-1.94,n_estimators=100,min_sum_hessian_in_leaf=(10**-4.2),min_child_samples =138,

                   colsample_bytree = 0.38,reg_lambda=10**1.35,random_state=1234,n_jobs=4)

lgbm_f=LGBMRegressor(num_leaves = 27,learning_rate =10**-1.79,n_estimators=100,min_sum_hessian_in_leaf=(10**-4.69),min_child_samples =54,subsample =0.8,subsample_freq=2,

                   colsample_bytree = 0.76,reg_lambda=10**1.08,random_state=1234,n_jobs=4)



from sklearn.preprocessing import OrdinalEncoder

oe=OrdinalEncoder()

X=oe.fit_transform(train[['Country_Region','Province_State']].fillna(''))

train['CR']=X[:,0]

train['PS']=X[:,1]



lgbm_cc.fit(train.loc[:,['LDConfirmedCases1','LDConfirmedCases2','LDConfirmedCases3','LDConfirmedCases4','LDConfirmedCases5','LDConfirmedCases6','LDConfirmedCases7','LDConfirmedCases8',

                                      'LDFatalities1','LDFatalities2','LDFatalities3','LDFatalities4','LDFatalities5','LDFatalities6','LDFatalities7','LDFatalities8',

                                         'CR','PS',

                                     'LDConfirmedCasesMA1','LDConfirmedCasesMA2','LDConfirmedCasesMA3','LDConfirmedCasesMA4','LDConfirmedCasesMA5','LDConfirmedCasesMA6','LDConfirmedCasesMA7','LDConfirmedCasesMA8',

                                     'LDFatalitiesMA1','LDFatalitiesMA2','LDFatalitiesMA3','LDFatalitiesMA4','LDFatalitiesMA5','LDFatalitiesMA6','LDFatalitiesMA7','LDFatalitiesMA8']],train.LDConfirmedCases,categorical_feature=['CR','PS'])

lgbm_f.fit(train.loc[:,['LDConfirmedCases1','LDConfirmedCases2','LDConfirmedCases3','LDConfirmedCases4','LDConfirmedCases5','LDConfirmedCases6','LDConfirmedCases7','LDConfirmedCases8',

                                      'LDFatalities1','LDFatalities2','LDFatalities3','LDFatalities4','LDFatalities5','LDFatalities6','LDFatalities7','LDFatalities8',

                                         'CR','PS',

                                     'LDConfirmedCasesMA1','LDConfirmedCasesMA2','LDConfirmedCasesMA3','LDConfirmedCasesMA4','LDConfirmedCasesMA5','LDConfirmedCasesMA6','LDConfirmedCasesMA7','LDConfirmedCasesMA8',

                                     'LDFatalitiesMA1','LDFatalitiesMA2','LDFatalitiesMA3','LDFatalitiesMA4','LDFatalitiesMA5','LDFatalitiesMA6','LDFatalitiesMA7','LDFatalitiesMA8']],train.LDFatalities,categorical_feature=['CR','PS'])

from sklearn.metrics import mean_squared_log_error

trainpred = pd.concat((train,test[test.Date>max(train.Date)])).reset_index(drop=True)

trainpred.sort_values(['Country_Region','Province_State','Date'],inplace=True)



X=oe.transform(trainpred[['Country_Region','Province_State']].fillna(''))

trainpred['CR']=X[:,0]

trainpred['PS']=X[:,1]



trainpred['serd']=trainpred.groupby('CRPS').cumcount()

trainpred['LConfirmedCases']=np.log1p(trainpred['ConfirmedCases'])

trainpred['LFatalities']=np.log1p(trainpred['Fatalities'])

trainpred['LDConfirmedCases']=trainpred.groupby('CRPS')[['LConfirmedCases']].diff()

trainpred['LDFatalities']=trainpred.groupby('CRPS')[['LFatalities']].diff()

trainpred['LARIMApred']=np.log1p(trainpred['ARIMApred'])

trainpred['LFARIMApred']=np.log1p(trainpred['FARIMApred'])

trainpred['LDARIMApred']=trainpred.groupby('CRPS')[['LARIMApred']].diff()

trainpred['LDFARIMApred']=trainpred.groupby('CRPS')[['LFARIMApred']].diff()



for serd in range(train.serd.max()+1,trainpred.serd.max()+1):

    print(serd)

    for i in range(1,8,1):

        trainpred['LFatalities'+str(i)]=trainpred.groupby('CRPS')[['LFatalities']].shift(i)

        trainpred['LDFatalities'+str(i)]=trainpred.groupby('CRPS')[['LDFatalities']].shift(i)

        trainpred['LConfirmedCases'+str(i)]=trainpred.groupby('CRPS')[['LConfirmedCases']].shift(i)

        trainpred['LDConfirmedCases'+str(i)]=trainpred.groupby('CRPS')[['LDConfirmedCases']].shift(i)

    trainpred['LDConfirmedCasesMA']=(trainpred.LDConfirmedCases+trainpred.LDConfirmedCases1+trainpred.LDConfirmedCases2+trainpred.LDConfirmedCases3+trainpred.LDConfirmedCases4)/5

    trainpred['LDFatalitiesMA']=(trainpred.LDFatalities+trainpred.LDFatalities1+trainpred.LDFatalities2+trainpred.LDFatalities3+trainpred.LDFatalities4)/5



    for i in range(1,8,1):

        trainpred['LDFatalitiesMA'+str(i)]=trainpred.groupby('CRPS')[['LDFatalitiesMA']].shift(i)

        trainpred['LDConfirmedCasesMA'+str(i)]=trainpred.groupby('CRPS')[['LDConfirmedCasesMA']].shift(i)



    trainpred.loc[trainpred.serd==serd,'LDConfirmedCases']= lgbm_cc.predict(trainpred.loc[trainpred.serd==serd,['LDConfirmedCases1','LDConfirmedCases2','LDConfirmedCases3','LDConfirmedCases4','LDConfirmedCases5','LDConfirmedCases6','LDConfirmedCases7','LDConfirmedCases8',

                                      'LDFatalities1','LDFatalities2','LDFatalities3','LDFatalities4','LDFatalities5','LDFatalities6','LDFatalities7','LDFatalities8',

                                         'CR','PS',

                                     'LDConfirmedCasesMA1','LDConfirmedCasesMA2','LDConfirmedCasesMA3','LDConfirmedCasesMA4','LDConfirmedCasesMA5','LDConfirmedCasesMA6','LDConfirmedCasesMA7','LDConfirmedCasesMA8',

                                     'LDFatalitiesMA1','LDFatalitiesMA2','LDFatalitiesMA3','LDFatalitiesMA4','LDFatalitiesMA5','LDFatalitiesMA6','LDFatalitiesMA7','LDFatalitiesMA8']])

    trainpred.loc[(trainpred.serd==serd) & (trainpred.LDConfirmedCases<0),'LDConfirmedCases']=0

    trainpred.loc[trainpred.serd==serd,'LConfirmedCases']=trainpred.loc[trainpred.serd==serd,'LDConfirmedCases']+trainpred.loc[trainpred.serd==serd,'LConfirmedCases1']

    trainpred.loc[trainpred.serd==serd,'ConfirmedCases']=np.exp(trainpred.loc[trainpred.serd==serd,'LConfirmedCases'])-1



    trainpred.loc[trainpred.serd==serd,'LDFatalities']= lgbm_f.predict(trainpred.loc[trainpred.serd==serd,['LDConfirmedCases1','LDConfirmedCases2','LDConfirmedCases3','LDConfirmedCases4','LDConfirmedCases5','LDConfirmedCases6','LDConfirmedCases7','LDConfirmedCases8',

                                      'LDFatalities1','LDFatalities2','LDFatalities3','LDFatalities4','LDFatalities5','LDFatalities6','LDFatalities7','LDFatalities8',

                                         'CR','PS',

                                     'LDConfirmedCasesMA1','LDConfirmedCasesMA2','LDConfirmedCasesMA3','LDConfirmedCasesMA4','LDConfirmedCasesMA5','LDConfirmedCasesMA6','LDConfirmedCasesMA7','LDConfirmedCasesMA8',

                                     'LDFatalitiesMA1','LDFatalitiesMA2','LDFatalitiesMA3','LDFatalitiesMA4','LDFatalitiesMA5','LDFatalitiesMA6','LDFatalitiesMA7','LDFatalitiesMA8']])

    trainpred.loc[(trainpred.serd==serd) & (trainpred.LDFatalities<0),'LDFatalities']=0

    trainpred.loc[trainpred.serd==serd,'LFatalities']=trainpred.loc[trainpred.serd==serd,'LDFatalities']+trainpred.loc[trainpred.serd==serd,'LFatalities1']

    trainpred.loc[trainpred.serd==serd,'Fatalities']=np.exp(trainpred.loc[trainpred.serd==serd,'LFatalities'])-1



print(np.sqrt(mean_squared_log_error (train.ConfirmedCases[train.serd>=71],trainpred.ConfirmedCases[(trainpred.serd>=71) & (trainpred.serd<=max(train.serd))])))

print(np.sqrt(mean_squared_log_error (train.Fatalities[train.serd>=71],trainpred.Fatalities[(trainpred.serd>=71) & (trainpred.serd<=max(train.serd))])))
for serd in range(train.serd.max()+1,trainpred.serd.max()+1):

    print(serd)

    if serd<=max(train.serd):

        trainpred.loc[trainpred.serd==serd,'LConfirmedCases']= trainpred.loc[trainpred.serd==serd,'RLConfirmedCases']

    else:

        trainpred.loc[trainpred.serd==serd,'LConfirmedCases']= qty.inverse_transform(rc.predict(qtx.transform(trainpred.loc[trainpred.serd==serd,['LConfirmedCases','LARIMApred']])).reshape(-1,1))

    trainpred.loc[trainpred.LConfirmedCases<0,'LConfirmedCases']=0

    trainpred.loc[trainpred.serd==serd,'LDConfirmedCases']=trainpred.loc[trainpred.serd==serd,'LConfirmedCases']-trainpred.loc[trainpred.serd==serd,'LConfirmedCases1']

    trainpred.loc[trainpred.LDConfirmedCases<0,'LDConfirmedCases']=0

    trainpred.loc[trainpred.serd==serd,'ConfirmedCases']=np.exp(trainpred.loc[trainpred.serd==serd,'LConfirmedCases'])-1    



    if serd<=max(train.serd):

        trainpred.loc[trainpred.serd==serd,'LFatalities']= trainpred.loc[trainpred.serd==serd,'RLFatalities']

    else:

        trainpred.loc[trainpred.serd==serd,'LFatalities']= qty_f.inverse_transform(rc_f.predict(qtx_f.transform(trainpred.loc[trainpred.serd==serd,['LConfirmedCases','LARIMApred','LFatalities','LFARIMApred','LDConfirmedCases','LDFatalities']])).reshape(-1,1))

    trainpred.loc[trainpred.LFatalities<0,'LFatalities']=0

    trainpred.loc[trainpred.serd==serd,'Fatalities']=np.exp(trainpred.loc[trainpred.serd==serd,'LFatalities'])-1    



private_trainpred=trainpred.copy()

private_trainpred.loc[private_trainpred.Country_Region.str.contains('Israel'),['Date','ConfirmedCases','Fatalities']].round()
submission=pd.concat((public_trainpred[(public_trainpred.Date>=min(test.Date)) & (public_trainpred.Date<=max(train.Date))],private_trainpred[(private_trainpred.Date>max(train.Date))]),axis=0)[['ForecastId','ConfirmedCases','Fatalities']]

submission.ForecastId=submission.ForecastId.astype('int')

submission.sort_values('ForecastId',inplace=True)

submission.to_csv('submission.csv',index=False)
from lightgbm import LGBMRegressor

def opt_lgbm(p):

    print(p)

    lgbm_cc=LGBMRegressor(num_leaves = p[0],learning_rate =10**p[1],n_estimators=100,min_sum_hessian_in_leaf=(10**p[2]),min_child_samples =p[3],subsample =p[4],subsample_freq=p[5],

                       colsample_bytree = p[6],reg_lambda=10**p[7],random_state=1234,n_jobs=4)

    lgbm_f=LGBMRegressor(num_leaves = p[8],learning_rate =10**p[9],n_estimators=100,min_sum_hessian_in_leaf=(10**p[10]),min_child_samples =p[11],subsample =p[12],subsample_freq=p[13],

                       colsample_bytree = p[14],reg_lambda=10**p[15],random_state=1234,n_jobs=4)



    lgbm_cc.fit(train.loc[train.serd<=70,['LDConfirmedCases1','LDConfirmedCases2','LDConfirmedCases3','LDConfirmedCases4','LDConfirmedCases5','LDConfirmedCases6','LDConfirmedCases7','LDConfirmedCases8',

                                          'LDFatalities1','LDFatalities2','LDFatalities3','LDFatalities4','LDFatalities5','LDFatalities6','LDFatalities7','LDFatalities8',

                                             'CR','PS',

                                         'LDConfirmedCasesMA1','LDConfirmedCasesMA2','LDConfirmedCasesMA3','LDConfirmedCasesMA4','LDConfirmedCasesMA5','LDConfirmedCasesMA6','LDConfirmedCasesMA7','LDConfirmedCasesMA8',

                                         'LDFatalitiesMA1','LDFatalitiesMA2','LDFatalitiesMA3','LDFatalitiesMA4','LDFatalitiesMA5','LDFatalitiesMA6','LDFatalitiesMA7','LDFatalitiesMA8']],train.LDConfirmedCases[train.serd<=70],categorical_feature=['CR','PS'])

    lgbm_f.fit(train.loc[train.serd<=70,['LDConfirmedCases1','LDConfirmedCases2','LDConfirmedCases3','LDConfirmedCases4','LDConfirmedCases5','LDConfirmedCases6','LDConfirmedCases7','LDConfirmedCases8',

                                          'LDFatalities1','LDFatalities2','LDFatalities3','LDFatalities4','LDFatalities5','LDFatalities6','LDFatalities7','LDFatalities8',

                                             'CR','PS',

                                         'LDConfirmedCasesMA1','LDConfirmedCasesMA2','LDConfirmedCasesMA3','LDConfirmedCasesMA4','LDConfirmedCasesMA5','LDConfirmedCasesMA6','LDConfirmedCasesMA7','LDConfirmedCasesMA8',

                                         'LDFatalitiesMA1','LDFatalitiesMA2','LDFatalitiesMA3','LDFatalitiesMA4','LDFatalitiesMA5','LDFatalitiesMA6','LDFatalitiesMA7','LDFatalitiesMA8']],train.LDFatalities[train.serd<=70],categorical_feature=['CR','PS'])

    trainpred = train.copy()

    for serd in range(71,max(train.serd)+1):

        for i in range(1,8,1):

            trainpred['LFatalities'+str(i)]=trainpred.groupby('CRPS')[['LFatalities']].shift(i)

            trainpred['LDFatalities'+str(i)]=trainpred.groupby('CRPS')[['LDFatalities']].shift(i)

            trainpred['LConfirmedCases'+str(i)]=trainpred.groupby('CRPS')[['LConfirmedCases']].shift(i)

            trainpred['LDConfirmedCases'+str(i)]=trainpred.groupby('CRPS')[['LDConfirmedCases']].shift(i)

        trainpred['LDConfirmedCasesMA']=(trainpred.LDConfirmedCases+trainpred.LDConfirmedCases1+trainpred.LDConfirmedCases2+trainpred.LDConfirmedCases3+trainpred.LDConfirmedCases4)/5

        trainpred['LDFatalitiesMA']=(trainpred.LDFatalities+trainpred.LDFatalities1+trainpred.LDFatalities2+trainpred.LDFatalities3+trainpred.LDFatalities4)/5



        for i in range(1,8,1):

            trainpred['LDFatalitiesMA'+str(i)]=trainpred.groupby('CRPS')[['LDFatalitiesMA']].shift(i)

            trainpred['LDConfirmedCasesMA'+str(i)]=trainpred.groupby('CRPS')[['LDConfirmedCasesMA']].shift(i)



        trainpred.loc[trainpred.serd==serd,'LDConfirmedCases']= lgbm_cc.predict(trainpred.loc[trainpred.serd==serd,['LDConfirmedCases1','LDConfirmedCases2','LDConfirmedCases3','LDConfirmedCases4','LDConfirmedCases5','LDConfirmedCases6','LDConfirmedCases7','LDConfirmedCases8',

                                          'LDFatalities1','LDFatalities2','LDFatalities3','LDFatalities4','LDFatalities5','LDFatalities6','LDFatalities7','LDFatalities8',

                                             'CR','PS',

                                         'LDConfirmedCasesMA1','LDConfirmedCasesMA2','LDConfirmedCasesMA3','LDConfirmedCasesMA4','LDConfirmedCasesMA5','LDConfirmedCasesMA6','LDConfirmedCasesMA7','LDConfirmedCasesMA8',

                                         'LDFatalitiesMA1','LDFatalitiesMA2','LDFatalitiesMA3','LDFatalitiesMA4','LDFatalitiesMA5','LDFatalitiesMA6','LDFatalitiesMA7','LDFatalitiesMA8']])

        trainpred.loc[(trainpred.serd==serd) & (trainpred.LDConfirmedCases<0),'LDConfirmedCases']=0

        trainpred.loc[trainpred.serd==serd,'LConfirmedCases']=trainpred.loc[trainpred.serd==serd,'LDConfirmedCases']+trainpred.loc[trainpred.serd==serd,'LConfirmedCases1']

        trainpred.loc[trainpred.serd==serd,'ConfirmedCases']=np.exp(trainpred.loc[trainpred.serd==serd,'LConfirmedCases'])-1



        trainpred.loc[trainpred.serd==serd,'LDFatalities']= lgbm_f.predict(trainpred.loc[trainpred.serd==serd,['LDConfirmedCases1','LDConfirmedCases2','LDConfirmedCases3','LDConfirmedCases4','LDConfirmedCases5','LDConfirmedCases6','LDConfirmedCases7','LDConfirmedCases8',

                                          'LDFatalities1','LDFatalities2','LDFatalities3','LDFatalities4','LDFatalities5','LDFatalities6','LDFatalities7','LDFatalities8',

                                             'CR','PS',

                                         'LDConfirmedCasesMA1','LDConfirmedCasesMA2','LDConfirmedCasesMA3','LDConfirmedCasesMA4','LDConfirmedCasesMA5','LDConfirmedCasesMA6','LDConfirmedCasesMA7','LDConfirmedCasesMA8',

                                         'LDFatalitiesMA1','LDFatalitiesMA2','LDFatalitiesMA3','LDFatalitiesMA4','LDFatalitiesMA5','LDFatalitiesMA6','LDFatalitiesMA7','LDFatalitiesMA8']])

        trainpred.loc[(trainpred.serd==serd) & (trainpred.LDFatalities<0),'LDFatalities']=0

        trainpred.loc[trainpred.serd==serd,'LFatalities']=trainpred.loc[trainpred.serd==serd,'LDFatalities']+trainpred.loc[trainpred.serd==serd,'LFatalities1']

        trainpred.loc[trainpred.serd==serd,'Fatalities']=np.exp(trainpred.loc[trainpred.serd==serd,'LFatalities'])-1



    ret=(np.sqrt(mean_squared_log_error (train.ConfirmedCases[train.serd>80],trainpred.ConfirmedCases[trainpred.serd>80]))+np.sqrt(mean_squared_log_error (train.Fatalities[train.serd>80],trainpred.Fatalities[train.serd>80])))/2

    

    print(ret)

    return ret



from skopt import gbrt_minimize,gp_minimize

ret=gbrt_minimize(opt_lgbm,[[5,80],[-4.0,0],[-5.0,-2],[1,200],[0.5,1.0],[1,20],[0.2,1.0],[-3.0,3.0],[5,80],[-4.0,0],[-5.0,-2],[1,200],[0.5,1.0],[1,20],[0.2,1.0],[-3.0,3.0]],n_calls=1,n_random_starts=1)

ret.fun
ret.x