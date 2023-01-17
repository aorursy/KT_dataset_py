import datetime

import numpy as np

import pandas as pd

import seaborn as sns

import xgboost as xgb

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

%matplotlib inline
def LeaveOneOut(data1, data2, groupcolumns, columnName, useLOO=False, cut=1, addNoise=False):

    features = list([])

    for a in groupcolumns:

        features.append(a)

    if(columnName is not None):

        features.append(columnName)

       

    grpCount = data1.groupby(features)['visitors'].count().reset_index().rename(columns={'visitors': 'Count'})

    if(useLOO):

        grpCount = grpCount[(grpCount.Count > cut)]

    grpMean = data1.groupby(features)['visitors'].mean().reset_index().rename(columns={'visitors': 'Mean'})

    grpMedian = data1.groupby(features)['visitors'].median().reset_index().rename(columns={'visitors': 'Median'})

    grpMin = data1.groupby(features)['visitors'].min().reset_index().rename(columns={'visitors': 'Min'})

    grpMax = data1.groupby(features)['visitors'].max().reset_index().rename(columns={'visitors': 'Max'})

    grpStd = data1.groupby(features)['visitors'].std().reset_index().rename(columns={'visitors': 'Std'})

        

    grpOutcomes = grpCount.merge(grpMean, on=features)

    grpOutcomes = grpOutcomes.merge(grpMedian, on=features)

    grpOutcomes = grpOutcomes.merge(grpMin, on=features)

    grpOutcomes = grpOutcomes.merge(grpMax, on=features)

    grpOutcomes = grpOutcomes.merge(grpStd, on=features)

    

    x = pd.merge(data2[features], grpOutcomes,

                 suffixes=('x_', ''),

                 how='left',

                 on=features,

                 left_index=True)[['Count','Mean','Median','Max','Min','Std']]

    x['Outcomes'] = data2['visitors'].values

    

    if(useLOO):

        nonnulls = (~x.Count.isnull())

        x.loc[nonnulls,'Mean'] = ((x[nonnulls].Mean*x[nonnulls].Count)-x[nonnulls].Outcomes)

        x.loc[nonnulls,'Median'] = ((x[nonnulls].Median*x[nonnulls].Count)-x[nonnulls].Outcomes)

        if(addNoise is True):

            x.loc[nonnulls&(x.Std>0),'Mean'] += np.random.normal(0,x[nonnulls&(x.Std>0)].Std,x[nonnulls&(x.Std>0)].shape[0])

            x.loc[nonnulls&(x.Std>0),'Median'] += np.random.normal(0,x[nonnulls&(x.Std>0)].Std,x[nonnulls&(x.Std>0)].shape[0])

        else:

            x.loc[nonnulls,'Count'] -= 1

        x.loc[nonnulls,'Mean'] /=  (x[nonnulls].Count)

        x.loc[nonnulls,'Median'] /= (x[nonnulls].Count)

    x.Count = np.log1p(x.Count)

    x = x.replace(np.inf, np.nan)

    x = x.replace(-np.inf, np.nan)

    #x = x.fillna(x.mean()) 

    

    return x[['Count','Mean','Median','Max','Min', 'Std']]





def MungeTrain():

    air_visit_data = pd.read_csv('../input/air_visit_data.csv',parse_dates=['visit_date'])

    air_store_info = pd.read_csv('../input/air_store_info.csv')

    

    hpg_store_info = pd.read_csv('../input/hpg_store_info.csv')

    hpg_store_info.drop(['latitude','longitude'],inplace=True,axis=1)

    store_id_relation = pd.read_csv('../input/store_id_relation.csv')

    storeinfo = air_store_info.merge(store_id_relation,on='air_store_id',how='left')

    storeinfo = storeinfo.merge(hpg_store_info,on='hpg_store_id',how='left')

    air_reserve = pd.read_csv('../input/air_reserve.csv',parse_dates=['visit_datetime'])

    air_reserve['visit_date'] = air_reserve.visit_datetime.apply( lambda df : 

    datetime.datetime(year=df.year, month=df.month, day=df.day))

    hpg_reserve =pd.read_csv('../input/hpg_reserve.csv',parse_dates=['visit_datetime'])

    hpg_reserve['visit_date'] = hpg_reserve.visit_datetime.apply( lambda df : 

    datetime.datetime(year=df.year, month=df.month, day=df.day))

    date_info = pd.read_csv('../input/date_info.csv',parse_dates=['calendar_date']).rename(columns={'calendar_date':'visit_date'})

    air_reserve_by_date = air_reserve.groupby(['air_store_id','visit_date']).reserve_visitors.sum().reset_index(drop=False)

    hpg_reserve_by_date = hpg_reserve.groupby(['hpg_store_id','visit_date']).reserve_visitors.sum().reset_index(drop=False)

    

    train = air_visit_data.merge(storeinfo,on='air_store_id',how='left')

    train = train.merge(air_reserve_by_date,on=['air_store_id','visit_date'],how='left')

    train = train.merge(hpg_reserve_by_date,on=['hpg_store_id','visit_date'],how='left')

    train = train.merge(date_info,on='visit_date',how='left')

    train['year'] = train.visit_date.dt.year

    train['month'] = train.visit_date.dt.month

    train.reserve_visitors_x = train.reserve_visitors_x.fillna(0)

    train.reserve_visitors_y = train.reserve_visitors_y.fillna(0)

    train.reserve_visitors_x = np.log1p(train.reserve_visitors_x)

    train.reserve_visitors_y = np.log1p(train.reserve_visitors_y)

    train.visitors = np.log1p(train.visitors)

    #train.drop(['latitude','longitude'],inplace=True,axis=1)

    train = train.fillna(-1)

    train = train.sort_values(by=['visit_date','air_store_id'],ascending=False)

    train = train.reset_index(drop=True)

    return train





def MungeTest(columns):

    air_visit_data = pd.read_csv('../input/sample_submission.csv')

    air_visit_data['visit_date'] = air_visit_data.id.apply(lambda x: datetime.datetime(year=int(x[-10:-6]), month=int(x[-5:-3]), day=int(x[-2:])))

    air_visit_data['air_store_id'] = air_visit_data.id.apply(lambda x: x[:-11])

    

    air_store_info = pd.read_csv('../input/air_store_info.csv')

    hpg_store_info = pd.read_csv('../input/hpg_store_info.csv')

    hpg_store_info.drop(['latitude','longitude'],inplace=True,axis=1)

    store_id_relation = pd.read_csv('../input/store_id_relation.csv')

    storeinfo = air_store_info.merge(store_id_relation,on='air_store_id',how='left')

    storeinfo = storeinfo.merge(hpg_store_info,on='hpg_store_id',how='left')

    air_reserve = pd.read_csv('../input/air_reserve.csv',parse_dates=['visit_datetime'])

    air_reserve['visit_date'] = air_reserve.visit_datetime.apply( lambda df : 

    datetime.datetime(year=df.year, month=df.month, day=df.day))

    hpg_reserve =pd.read_csv('../input/hpg_reserve.csv',parse_dates=['visit_datetime'])

    hpg_reserve['visit_date'] = hpg_reserve.visit_datetime.apply( lambda df : 

    datetime.datetime(year=df.year, month=df.month, day=df.day))

    date_info = pd.read_csv('../input/date_info.csv',parse_dates=['calendar_date']).rename(columns={'calendar_date':'visit_date'})

    air_reserve_by_date = air_reserve.groupby(['air_store_id','visit_date']).reserve_visitors.sum().reset_index(drop=False)

    hpg_reserve_by_date = hpg_reserve.groupby(['hpg_store_id','visit_date']).reserve_visitors.sum().reset_index(drop=False)

    

    test = air_visit_data.merge(storeinfo,on='air_store_id',how='left')

    test = test.merge(air_reserve_by_date,on=['air_store_id','visit_date'],how='left')

    test = test.merge(hpg_reserve_by_date,on=['hpg_store_id','visit_date'],how='left')

    test = test.merge(date_info,on='visit_date',how='left')

    test['year'] = test.visit_date.dt.year

    test['month'] = test.visit_date.dt.month

    test.reserve_visitors_x = test.reserve_visitors_x.fillna(0)

    test.reserve_visitors_y = test.reserve_visitors_y.fillna(0)

    test.reserve_visitors_x = np.log1p(test.reserve_visitors_x)

    test.reserve_visitors_y = np.log1p(test.reserve_visitors_y)

    test.visitors = np.log1p(test.visitors)

    #test.drop(['latitude','longitude'],inplace=True,axis=1)

    test = test.fillna(-1)

    test = test.sort_values(by=['visit_date','air_store_id'],ascending=False)

    test = test.reset_index(drop=True)

    return test[list(['id'])+list(columns)]
train = MungeTrain()

delta = train.visit_date.max()-pd.Timedelta(weeks=5)

lastfiveweekstrain = train[train.visit_date>=delta].copy()

lastfiveweekstrain = lastfiveweekstrain.reset_index(drop=True)

train = train[train.visit_date<delta].copy()

train = train.reset_index(drop=True)

test = MungeTest(train.columns)
# Time to do predictions for 60 weeks total

deltavals = [1,2,3,4,5]

xgbtrainpreds = None

xgbtestpreds = None

for i,deltaweek in enumerate(deltavals):

    print(i)

    delta = pd.Timedelta(weeks=deltaweek)

    blindmin = train.visit_date.max()-delta

    blindmax = train.visit_date.max()

    vismin = blindmin-delta-pd.Timedelta(days=1)

    vismax = blindmax-delta-pd.Timedelta(days=1)

    btrain = None



    for x in range(int(60./deltaweek)):

        vistrain = train[(train.visit_date<=vismax)].copy()

        blindtrain = train[(train.visit_date>=blindmin)&(train.visit_date<=blindmax)].copy()

        features = ['air_genre_name',

                    'air_area_name', 'hpg_store_id',

                    'hpg_genre_name', 'hpg_area_name', 

                    'day_of_week', 'holiday_flg']

        for c in features:

            blindtrain[c+'_Count_Store'] = np.nan

            blindtrain[c+'_Mean_Store'] = np.nan

            blindtrain[c+'_Median_Store'] = np.nan

            blindtrain[c+'_Max_Store'] = np.nan

            blindtrain[c+'_Min_Store'] = np.nan

            blindtrain[c+'_Std_Store'] = np.nan





            blindtrain[[c+'_Count_Store',c+'_Mean_Store',

                        c+'_Median_Store',c+'_Max_Store',

                        c+'_Min_Store', c+'_Std_Store']] =  LeaveOneOut(vistrain,

                                                                        blindtrain,

                                                                        list(['air_store_id']),

                                                                        c,

                                                                        useLOO=False,

                                                                        cut=0).values



        features = ['air_store_id',

                    'air_genre_name',

                    'air_area_name', 'hpg_store_id',

                    'hpg_genre_name', 'hpg_area_name', 

                    'day_of_week', 'holiday_flg']



        for c in features:

            blindtrain[c+'_Count'] = np.nan

            blindtrain[c+'_Mean'] = np.nan

            blindtrain[c+'_Median'] = np.nan

            blindtrain[c+'_Max'] = np.nan

            blindtrain[c+'_Min'] = np.nan

            blindtrain[c+'_Std'] = np.nan



            blindtrain[[c+'_Count',c+'_Mean',

                        c+'_Median',c+'_Max',

                        c+'_Min', c+'_Std']] =  LeaveOneOut(vistrain,

                                                            blindtrain,

                                                            list([]),

                                                            c,

                                                            useLOO=False,

                                                            cut=0,

                                                            addNoise=False).values





            if('air_store_id'!=c):

                blindtrain.drop(c,inplace=True,axis=1)

        if(btrain is None):

            btrain = blindtrain.copy()

        else:

            btrain = pd.concat([btrain,blindtrain])

        vismax -= pd.Timedelta(weeks=deltaweek)

        vismin -= pd.Timedelta(weeks=deltaweek)

        blindmin -= pd.Timedelta(weeks=deltaweek)

        blindmax -= pd.Timedelta(weeks=deltaweek)



    vistrain = train.copy()

    features = ['air_genre_name',

                'air_area_name', 'hpg_store_id',

                'hpg_genre_name', 'hpg_area_name', 

                'day_of_week', 'holiday_flg']

    for c in features:

        lastfiveweekstrain[c+'_Count_Store'] = np.nan

        lastfiveweekstrain[c+'_Mean_Store'] = np.nan

        lastfiveweekstrain[c+'_Median_Store'] = np.nan

        lastfiveweekstrain[c+'_Max_Store'] = np.nan

        lastfiveweekstrain[c+'_Min_Store'] = np.nan

        lastfiveweekstrain[c+'_Std_Store'] = np.nan





        lastfiveweekstrain[[c+'_Count_Store',c+'_Mean_Store',

                            c+'_Median_Store',c+'_Max_Store',

                            c+'_Min_Store', c+'_Std_Store']] =  LeaveOneOut(vistrain,

                                                                            lastfiveweekstrain,

                                                                            list(['air_store_id']),

                                                                            c,

                                                                            useLOO=False,

                                                                            cut=0).values

        

        test[c+'_Count_Store'] = np.nan

        test[c+'_Mean_Store'] = np.nan

        test[c+'_Median_Store'] = np.nan

        test[c+'_Max_Store'] = np.nan

        test[c+'_Min_Store'] = np.nan

        test[c+'_Std_Store'] = np.nan





        test[[c+'_Count_Store',c+'_Mean_Store',

              c+'_Median_Store',c+'_Max_Store',

              c+'_Min_Store', c+'_Std_Store']] =  LeaveOneOut(vistrain,

                                                              test,

                                                              list(['air_store_id']),

                                                              c,

                                                              useLOO=False,

                                                              cut=0).values



    features = ['air_store_id',

                'air_genre_name',

                'air_area_name', 'hpg_store_id',

                'hpg_genre_name', 'hpg_area_name', 

                'day_of_week', 'holiday_flg']



    for c in features:

        lastfiveweekstrain[c+'_Count'] = np.nan

        lastfiveweekstrain[c+'_Mean'] = np.nan

        lastfiveweekstrain[c+'_Median'] = np.nan

        lastfiveweekstrain[c+'_Max'] = np.nan

        lastfiveweekstrain[c+'_Min'] = np.nan

        lastfiveweekstrain[c+'_Std'] = np.nan



        lastfiveweekstrain[[c+'_Count',c+'_Mean',

                            c+'_Median',c+'_Max',

                            c+'_Min', c+'_Std']] =  LeaveOneOut(vistrain,

                                                                lastfiveweekstrain,

                                                                list([]),

                                                                c,

                                                                useLOO=False,

                                                                cut=0,

                                                                addNoise=False).values

        

        test[c+'_Count'] = np.nan

        test[c+'_Mean'] = np.nan

        test[c+'_Median'] = np.nan

        test[c+'_Max'] = np.nan

        test[c+'_Min'] = np.nan

        test[c+'_Std'] = np.nan



        test[[c+'_Count',c+'_Mean',

              c+'_Median',c+'_Max',

              c+'_Min', c+'_Std']] =  LeaveOneOut(vistrain,

                                                  test,

                                                  list([]),

                                                  c,

                                                  useLOO=False,

                                                  cut=0,

                                                  addNoise=False).values

    

    d_train = xgb.DMatrix(btrain[btrain.columns[3:]], label=btrain.visitors)

    d_valid = xgb.DMatrix(lastfiveweekstrain[btrain.columns[3:]], label=lastfiveweekstrain.visitors)

    d_test = xgb.DMatrix(test[btrain.columns[3:]])

    params = {}

    params['objective'] = 'reg:linear'

    params['eval_metric'] = 'rmse'

    params['eta'] = 0.1

    params['max_depth'] = 7

    params['subsample']= 0.8 

    params['colsample_bytree']= 0.8

    params['silent'] = 1

    watchlist = [(d_train, 'train'), (d_valid, 'valid')]

    clf = xgb.train(params, d_train, 1000, watchlist, early_stopping_rounds=20, verbose_eval=10)

    xgbx = clf.predict(d_valid,ntree_limit=clf.best_iteration+1)

    xgby = clf.predict(d_test,ntree_limit=clf.best_iteration+1)

    trainpreds = pd.DataFrame()

    trainpreds['visit_date'] = lastfiveweekstrain.visit_date.values

    trainpreds['air_store_id'] = lastfiveweekstrain.air_store_id.values

    trainpreds['visitors'] = lastfiveweekstrain.visitors.values

    trainpreds['weeks_'+str(deltaweek)] = xgbx

    testpreds = pd.DataFrame()

    testpreds['visit_date'] = test.visit_date.values

    testpreds['air_store_id'] = test.air_store_id.values

    testpreds['weeks_'+str(deltaweek)] = xgby

    if(xgbtrainpreds is None):

        xgbtrainpreds = trainpreds.copy()

        xgbtestpreds = testpreds.copy()

    else:

        xgbtrainpreds = xgbtrainpreds.merge(trainpreds[['air_store_id','visit_date','weeks_'+str(deltaweek)]],on=['air_store_id','visit_date'])

        xgbtestpreds = xgbtestpreds.merge(testpreds[['air_store_id','visit_date','weeks_'+str(deltaweek)]],on=['air_store_id','visit_date'])
xgbtrainpreds.head()
xgbtestpreds.head()
X_train, X_test, y_train, y_test = train_test_split(xgbtrainpreds[['weeks_1', 'weeks_2', 'weeks_3', 'weeks_4', 'weeks_5']],

                                                    xgbtrainpreds.visitors, test_size=0.2, random_state=42)

d_train = xgb.DMatrix(X_train, label=y_train)

d_valid = xgb.DMatrix(X_test, label=y_test)

d_test = xgb.DMatrix(xgbtestpreds[['weeks_1', 'weeks_2', 'weeks_3', 'weeks_4', 'weeks_5']])

params = {}

params['objective'] = 'reg:linear'

params['eval_metric'] = 'rmse'

params['eta'] = 0.2

params['max_depth'] = 3

params['subsample']=0.8 

params['colsample_bytree']=0.8

params['silent'] = 1

watchlist = [(d_train, 'train'), (d_valid, 'valid')]

clf = xgb.train(params, d_train, 250, watchlist, early_stopping_rounds=20, verbose_eval=10)

xgbx = clf.predict(d_valid,ntree_limit=clf.best_iteration+1)

xgby = clf.predict(d_test,ntree_limit=clf.best_iteration+1)
bestrounds = clf.best_iteration+1

d_train = xgb.DMatrix(xgbtrainpreds[['weeks_1', 'weeks_2', 'weeks_3', 'weeks_4', 'weeks_5']], label=xgbtrainpreds.visitors)

clf = xgb.train(params, d_train, int((bestrounds)*1.2), verbose_eval=10)

xgbx = clf.predict(d_train,ntree_limit=bestrounds)

xgby = clf.predict(d_test,ntree_limit=bestrounds)
print(np.sqrt(mean_squared_error(xgbtrainpreds.visitors.ravel(),

                                 xgbx)))
xgbtestpreds['id'] =  xgbtestpreds["air_store_id"]+'_'+xgbtestpreds["visit_date"].map(str)

xgbtestpreds['id'] = xgbtestpreds.id.str[:-9]

xgbtestpreds['visitors'] = np.expm1(xgby)
xgbtestpreds[['id','visitors']].to_csv('xgbtimeseries.csv',index=False)