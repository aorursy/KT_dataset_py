import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

train = pd.read_csv ('/kaggle/input/covid19-global-forecasting-week-4/train.csv')

test = pd.read_csv ('/kaggle/input/covid19-global-forecasting-week-4/test.csv')

train['CRPS']=train.Country_Region+train.Province_State.fillna('')

test['CRPS']=test.Country_Region+test.Province_State.fillna('')

train
train['LConfirmedCases']=np.log1p(train['ConfirmedCases'])

train['LFatalities']=np.log1p(train['Fatalities'])

train['LDConfirmedCases']=train.groupby('CRPS')[['LConfirmedCases']].diff()

train['LDFatalities']=train.groupby('CRPS')[['LFatalities']].diff()



train['LConfirmedCases1']=train.groupby('CRPS')[['LConfirmedCases']].shift(1)

train['LConfirmedCases2']=train.groupby('CRPS')[['LConfirmedCases']].shift(2)

train['LConfirmedCases3']=train.groupby('CRPS')[['LConfirmedCases']].shift(3)

train['LConfirmedCases4']=train.groupby('CRPS')[['LConfirmedCases']].shift(4)

train['LDConfirmedCases1']=train.groupby('CRPS')[['LDConfirmedCases']].shift(1)

train['LDConfirmedCases2']=train.groupby('CRPS')[['LDConfirmedCases']].shift(2)

train['LDConfirmedCases3']=train.groupby('CRPS')[['LDConfirmedCases']].shift(3)

train['LDConfirmedCases4']=train.groupby('CRPS')[['LDConfirmedCases']].shift(4)

train['LDConfirmedCasesMA']=(train.LDConfirmedCases+train.LDConfirmedCases1+train.LDConfirmedCases2+train.LDConfirmedCases3+train.LDConfirmedCases4)/5

train['LDConfirmedCasesMA1']=train.groupby('CRPS')[['LDConfirmedCasesMA']].shift(1)

train['LDConfirmedCasesMA2']=train.groupby('CRPS')[['LDConfirmedCasesMA']].shift(2)

train['LDConfirmedCasesMA3']=train.groupby('CRPS')[['LDConfirmedCasesMA']].shift(3)

train['LDConfirmedCasesMA4']=train.groupby('CRPS')[['LDConfirmedCasesMA']].shift(4)

train['LConfirmedCasesMA']=(train.LConfirmedCases+train.LConfirmedCases1+train.LConfirmedCases2+train.LConfirmedCases3+train.LConfirmedCases4)/5

train['LConfirmedCasesMA1']=train.groupby('CRPS')[['LConfirmedCasesMA']].shift(1)

train['LConfirmedCasesMA2']=train.groupby('CRPS')[['LConfirmedCasesMA']].shift(2)

train['LConfirmedCasesMA3']=train.groupby('CRPS')[['LConfirmedCasesMA']].shift(3)

train['LConfirmedCasesMA4']=train.groupby('CRPS')[['LConfirmedCasesMA']].shift(4)



train['LFatalities1']=train.groupby('CRPS')[['LFatalities']].shift(1)

train['LFatalities2']=train.groupby('CRPS')[['LFatalities']].shift(2)

train['LFatalities3']=train.groupby('CRPS')[['LFatalities']].shift(3)

train['LFatalities4']=train.groupby('CRPS')[['LFatalities']].shift(4)

train['LDFatalities1']=train.groupby('CRPS')[['LDFatalities']].shift(1)

train['LDFatalities2']=train.groupby('CRPS')[['LDFatalities']].shift(2)

train['LDFatalities3']=train.groupby('CRPS')[['LDFatalities']].shift(3)

train['LDFatalities4']=train.groupby('CRPS')[['LDFatalities']].shift(4)

train['LDFatalitiesMA']=(train.LDFatalities+train.LDFatalities1+train.LDFatalities2+train.LDFatalities3+train.LDFatalities4)/5

train['LDFatalitiesMA1']=train.groupby('CRPS')[['LDFatalitiesMA']].shift(1)

train['LDFatalitiesMA2']=train.groupby('CRPS')[['LDFatalitiesMA']].shift(2)

train['LDFatalitiesMA3']=train.groupby('CRPS')[['LDFatalitiesMA']].shift(3)

train['LDFatalitiesMA4']=train.groupby('CRPS')[['LDFatalitiesMA']].shift(4)

train['LFatalitiesMA']=(train.LFatalities+train.LFatalities1+train.LFatalities2+train.LFatalities3+train.LFatalities4)/5

train['LFatalitiesMA1']=train.groupby('CRPS')[['LFatalitiesMA']].shift(1)

train['LFatalitiesMA2']=train.groupby('CRPS')[['LFatalitiesMA']].shift(2)

train['LFatalitiesMA3']=train.groupby('CRPS')[['LFatalitiesMA']].shift(3)

train['LFatalitiesMA4']=train.groupby('CRPS')[['LFatalitiesMA']].shift(4)

train['serd']=train.groupby('CRPS').cumcount()

train.loc[train.ConfirmedCases==0,'days_since_confirmed']=0

train.loc[train.ConfirmedCases>0,'days_since_confirmed']=train[train.ConfirmedCases>0].groupby('CRPS').cumcount() #The first is 0 to avoid leakakge

from lightgbm import LGBMRegressor



lgbm_cc=LGBMRegressor(num_leaves = 85,learning_rate =10**-1.89,n_estimators=100,min_sum_hessian_in_leaf=(10**-4.1),min_child_samples =2,subsample =0.97,subsample_freq=10,

                   colsample_bytree = 0.68,reg_lambda=10**1.4,random_state=1234,n_jobs=4)

lgbm_f=LGBMRegressor(num_leaves = 26,learning_rate =10**-1.63,n_estimators=100,min_sum_hessian_in_leaf=(10**-4.04),min_child_samples =14,subsample =0.66,subsample_freq=5,

                   colsample_bytree = 0.8,reg_lambda=10**1.92,random_state=1234,n_jobs=4)



from sklearn.preprocessing import OrdinalEncoder

oe=OrdinalEncoder()

X=oe.fit_transform(train[['Country_Region','Province_State']].fillna(''))

train['CR']=X[:,0]

train['PS']=X[:,1]



lgbm_cc.fit(train.loc[:,['LDConfirmedCases1','LDConfirmedCases2','LDConfirmedCases3','LDConfirmedCases4',

                                  'LDFatalities1','LDFatalities2','LDFatalities3','LDFatalities4','days_since_confirmed','CR','PS',

                                 'LDConfirmedCasesMA1','LDConfirmedCasesMA2','LDConfirmedCasesMA3','LDConfirmedCasesMA4',

                                 'LDFatalitiesMA1','LDFatalitiesMA2','LDFatalitiesMA3','LDFatalitiesMA4',

                                     'LConfirmedCases1','LConfirmedCases2','LConfirmedCases3','LConfirmedCases4',

                                  'LFatalities1','LFatalities2','LFatalities3','LFatalities4',

                                 'LConfirmedCasesMA1','LConfirmedCasesMA2','LConfirmedCasesMA3','LConfirmedCasesMA4',

                                 'LFatalitiesMA1','LFatalitiesMA2','LFatalitiesMA3','LFatalitiesMA4']],train.LDConfirmedCases,categorical_feature=['CR','PS'])

lgbm_f.fit(train.loc[:,['LDConfirmedCases1','LDConfirmedCases2','LDConfirmedCases3','LDConfirmedCases4',

                                  'LDFatalities1','LDFatalities2','LDFatalities3','LDFatalities4','days_since_confirmed','CR','PS',

                                 'LDConfirmedCasesMA1','LDConfirmedCasesMA2','LDConfirmedCasesMA3','LDConfirmedCasesMA4',

                                 'LDFatalitiesMA1','LDFatalitiesMA2','LDFatalitiesMA3','LDFatalitiesMA4',

                                     'LConfirmedCases1','LConfirmedCases2','LConfirmedCases3','LConfirmedCases4',

                                  'LFatalities1','LFatalities2','LFatalities3','LFatalities4',

                                 'LConfirmedCasesMA1','LConfirmedCasesMA2','LConfirmedCasesMA3','LConfirmedCasesMA4',

                                 'LFatalitiesMA1','LFatalitiesMA2','LFatalitiesMA3','LFatalitiesMA4','LConfirmedCases','LDConfirmedCases']],train.LDFatalities,categorical_feature=['CR','PS'])    

from sklearn.metrics import mean_squared_log_error

train['serd']=train.groupby('CRPS').cumcount()

trainpred = pd.concat((train,test[test.Date>train.Date.max()])).reset_index(drop=True)

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



for serd in range(train.serd.max()+1,trainpred.serd.max()+1):

    print(serd)

    trainpred['LConfirmedCases1']=trainpred.groupby('CRPS')[['LConfirmedCases']].shift(1)

    trainpred['LConfirmedCases2']=trainpred.groupby('CRPS')[['LConfirmedCases']].shift(2)

    trainpred['LConfirmedCases3']=trainpred.groupby('CRPS')[['LConfirmedCases']].shift(3)

    trainpred['LConfirmedCases4']=trainpred.groupby('CRPS')[['LConfirmedCases']].shift(4)

    trainpred['LDConfirmedCases1']=trainpred.groupby('CRPS')[['LDConfirmedCases']].shift(1)

    trainpred['LDConfirmedCases2']=trainpred.groupby('CRPS')[['LDConfirmedCases']].shift(2)

    trainpred['LDConfirmedCases3']=trainpred.groupby('CRPS')[['LDConfirmedCases']].shift(3)

    trainpred['LDConfirmedCases4']=trainpred.groupby('CRPS')[['LDConfirmedCases']].shift(4)

    trainpred['LDConfirmedCasesMA']=(trainpred.LDConfirmedCases+trainpred.LDConfirmedCases1+trainpred.LDConfirmedCases2+trainpred.LDConfirmedCases3+trainpred.LDConfirmedCases4)/5

    trainpred['LDConfirmedCasesMA1']=trainpred.groupby('CRPS')[['LDConfirmedCasesMA']].shift(1)

    trainpred['LDConfirmedCasesMA2']=trainpred.groupby('CRPS')[['LDConfirmedCasesMA']].shift(2)

    trainpred['LDConfirmedCasesMA3']=trainpred.groupby('CRPS')[['LDConfirmedCasesMA']].shift(3)

    trainpred['LDConfirmedCasesMA4']=trainpred.groupby('CRPS')[['LDConfirmedCasesMA']].shift(4)

    trainpred['LConfirmedCasesMA']=(trainpred.LConfirmedCases+trainpred.LConfirmedCases1+trainpred.LConfirmedCases2+trainpred.LConfirmedCases3+trainpred.LConfirmedCases4)/5

    trainpred['LConfirmedCasesMA1']=trainpred.groupby('CRPS')[['LConfirmedCasesMA']].shift(1)

    trainpred['LConfirmedCasesMA2']=trainpred.groupby('CRPS')[['LConfirmedCasesMA']].shift(2)

    trainpred['LConfirmedCasesMA3']=trainpred.groupby('CRPS')[['LConfirmedCasesMA']].shift(3)

    trainpred['LConfirmedCasesMA4']=trainpred.groupby('CRPS')[['LConfirmedCasesMA']].shift(4)



    trainpred['LFatalities1']=trainpred.groupby('CRPS')[['LFatalities']].shift(1)

    trainpred['LFatalities2']=trainpred.groupby('CRPS')[['LFatalities']].shift(2)

    trainpred['LFatalities3']=trainpred.groupby('CRPS')[['LFatalities']].shift(3)

    trainpred['LFatalities4']=trainpred.groupby('CRPS')[['LFatalities']].shift(4)

    trainpred['LDFatalities1']=trainpred.groupby('CRPS')[['LDFatalities']].shift(1)

    trainpred['LDFatalities2']=trainpred.groupby('CRPS')[['LDFatalities']].shift(2)

    trainpred['LDFatalities3']=trainpred.groupby('CRPS')[['LDFatalities']].shift(3)

    trainpred['LDFatalities4']=trainpred.groupby('CRPS')[['LDFatalities']].shift(4)

    trainpred['LDFatalitiesMA']=(trainpred.LDFatalities+trainpred.LDFatalities1+trainpred.LDFatalities2+trainpred.LDFatalities3+trainpred.LDFatalities4)/5

    trainpred['LDFatalitiesMA1']=trainpred.groupby('CRPS')[['LDFatalitiesMA']].shift(1)

    trainpred['LDFatalitiesMA2']=trainpred.groupby('CRPS')[['LDFatalitiesMA']].shift(2)

    trainpred['LDFatalitiesMA3']=trainpred.groupby('CRPS')[['LDFatalitiesMA']].shift(3)

    trainpred['LDFatalitiesMA4']=trainpred.groupby('CRPS')[['LDFatalitiesMA']].shift(4)

    trainpred['LFatalitiesMA']=(trainpred.LFatalities+trainpred.LFatalities1+trainpred.LFatalities2+trainpred.LFatalities3+trainpred.LFatalities4)/5

    trainpred['LFatalitiesMA1']=trainpred.groupby('CRPS')[['LFatalitiesMA']].shift(1)

    trainpred['LFatalitiesMA2']=trainpred.groupby('CRPS')[['LFatalitiesMA']].shift(2)

    trainpred['LFatalitiesMA3']=trainpred.groupby('CRPS')[['LFatalitiesMA']].shift(3)

    trainpred['LFatalitiesMA4']=trainpred.groupby('CRPS')[['LFatalitiesMA']].shift(4)

    trainpred.loc[trainpred.serd==serd,'LDConfirmedCases']= lgbm_cc.predict(trainpred.loc[trainpred.serd==serd,['LDConfirmedCases1','LDConfirmedCases2','LDConfirmedCases3','LDConfirmedCases4',

                                  'LDFatalities1','LDFatalities2','LDFatalities3','LDFatalities4','days_since_confirmed','CR','PS',

                                 'LDConfirmedCasesMA1','LDConfirmedCasesMA2','LDConfirmedCasesMA3','LDConfirmedCasesMA4',

                                 'LDFatalitiesMA1','LDFatalitiesMA2','LDFatalitiesMA3','LDFatalitiesMA4',

                                     'LConfirmedCases1','LConfirmedCases2','LConfirmedCases3','LConfirmedCases4',

                                  'LFatalities1','LFatalities2','LFatalities3','LFatalities4',

                                 'LConfirmedCasesMA1','LConfirmedCasesMA2','LConfirmedCasesMA3','LConfirmedCasesMA4',

                                 'LFatalitiesMA1','LFatalitiesMA2','LFatalitiesMA3','LFatalitiesMA4']])

    trainpred.loc[(trainpred.serd==serd) & (trainpred.LDConfirmedCases<0),'LDConfirmedCases']=0

    trainpred.loc[trainpred.serd==serd,'LConfirmedCases']=trainpred.loc[trainpred.serd==serd,'LDConfirmedCases']+trainpred.loc[trainpred.serd==serd,'LConfirmedCases1']

    trainpred.loc[trainpred.serd==serd,'ConfirmedCases']=np.exp(trainpred.loc[trainpred.serd==serd,'LConfirmedCases'])-1

    

    trainpred.loc[trainpred.serd==serd,'LDFatalities']= lgbm_f.predict(trainpred.loc[trainpred.serd==serd,['LDConfirmedCases1','LDConfirmedCases2','LDConfirmedCases3','LDConfirmedCases4',

                                  'LDFatalities1','LDFatalities2','LDFatalities3','LDFatalities4','days_since_confirmed','CR','PS',

                                 'LDConfirmedCasesMA1','LDConfirmedCasesMA2','LDConfirmedCasesMA3','LDConfirmedCasesMA4',

                                 'LDFatalitiesMA1','LDFatalitiesMA2','LDFatalitiesMA3','LDFatalitiesMA4',

                                     'LConfirmedCases1','LConfirmedCases2','LConfirmedCases3','LConfirmedCases4',

                                  'LFatalities1','LFatalities2','LFatalities3','LFatalities4',

                                 'LConfirmedCasesMA1','LConfirmedCasesMA2','LConfirmedCasesMA3','LConfirmedCasesMA4',

                                 'LFatalitiesMA1','LFatalitiesMA2','LFatalitiesMA3','LFatalitiesMA4','LConfirmedCases','LDConfirmedCases']])

    trainpred.loc[(trainpred.serd==serd) & (trainpred.LDFatalities<0),'LDFatalities']=0

    trainpred.loc[trainpred.serd==serd,'LFatalities']=trainpred.loc[trainpred.serd==serd,'LDFatalities']+trainpred.loc[trainpred.serd==serd,'LFatalities1']

    trainpred.loc[trainpred.serd==serd,'Fatalities']=np.exp(trainpred.loc[trainpred.serd==serd,'LFatalities'])-1
trainpred.loc[(trainpred.Date<=max(test.Date)) & (trainpred.Date>=min(test.Date)),'ForecastId']=test.loc[:,'ForecastId'].values

submission=trainpred.loc[trainpred.Date>=min(test.Date)][['ForecastId','ConfirmedCases','Fatalities']]

submission.ForecastId=submission.ForecastId.astype('int')

submission.sort_values('ForecastId',inplace=True)

submission.to_csv('submission.csv',index=False)