import datetime

import numpy as np

import pandas as pd

from sklearn.metrics import mean_squared_error
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

    x = x.fillna(x.mean()) 

    

    return x[['Count','Mean','Median','Max','Min', 'Std']]





def MungeTrain():

    air_visit_data = pd.read_csv('../input/air_visit_data.csv',parse_dates=['visit_date'])

    air_store_info = pd.read_csv('../input/air_store_info.csv')

    hpg_store_info = pd.read_csv('../input/hpg_store_info.csv')

    air_reserve = pd.read_csv('../input/air_reserve.csv',parse_dates=['visit_datetime'])

    air_reserve['visit_date'] = air_reserve.visit_datetime.apply( lambda df : 

    datetime.datetime(year=df.year, month=df.month, day=df.day))

    hpg_reserve =pd.read_csv('../input/hpg_reserve.csv',parse_dates=['visit_datetime'])

    hpg_reserve['visit_date'] = hpg_reserve.visit_datetime.apply( lambda df : 

    datetime.datetime(year=df.year, month=df.month, day=df.day))

    store_id_relation = pd.read_csv('../input/store_id_relation.csv')

    date_info = pd.read_csv('../input/date_info.csv',parse_dates=['calendar_date']).rename(columns={'calendar_date':'visit_date'})

    air_reserve_by_date = air_reserve.groupby(['air_store_id','visit_date']).reserve_visitors.sum().reset_index(drop=False)

    hpg_reserve_by_date = hpg_reserve.groupby(['hpg_store_id','visit_date']).reserve_visitors.sum().reset_index(drop=False)

    hpg_store_info.drop(['latitude','longitude'],inplace=True,axis=1)

    train = air_visit_data.merge(air_store_info,on='air_store_id')

    train = train.merge(air_reserve_by_date,on=['air_store_id','visit_date'],how='left')

    train = train.merge(store_id_relation,on='air_store_id',how='left')

    train = train.merge(hpg_store_info,on='hpg_store_id',how='left')

    train = train.merge(hpg_reserve_by_date,on=['hpg_store_id','visit_date'],how='left')

    train = train.merge(date_info,on='visit_date',how='left')

    train['year'] = train.visit_date.dt.year

    train['month'] = train.visit_date.dt.month

    train.reserve_visitors_x = train.reserve_visitors_x.fillna(0)

    train.reserve_visitors_y = train.reserve_visitors_y.fillna(0)

    train.reserve_visitors_x = np.log1p(train.reserve_visitors_x)

    train.reserve_visitors_y = np.log1p(train.reserve_visitors_y)

    train.visitors = np.log1p(train.visitors)

    train.drop(['latitude','longitude'],inplace=True,axis=1)

    train = train.fillna(-1)

    train = train.sort_values(by='visit_date')

    return train





def MungeTest(columns):

    air_visit_data = pd.read_csv('../input/sample_submission.csv')

    air_visit_data['visit_date'] = air_visit_data.id.apply(lambda x: datetime.datetime(year=int(x[-10:-6]), month=int(x[-5:-3]), day=int(x[-2:])))

    air_visit_data['air_store_id'] = air_visit_data.id.apply(lambda x: x[:-11])

    air_store_info = pd.read_csv('../input/air_store_info.csv')

    hpg_store_info = pd.read_csv('../input/hpg_store_info.csv')

    air_reserve = pd.read_csv('../input/air_reserve.csv',parse_dates=['visit_datetime'])

    air_reserve['visit_date'] = air_reserve.visit_datetime.apply( lambda df : 

    datetime.datetime(year=df.year, month=df.month, day=df.day))

    hpg_reserve =pd.read_csv('../input/hpg_reserve.csv',parse_dates=['visit_datetime'])

    hpg_reserve['visit_date'] = hpg_reserve.visit_datetime.apply( lambda df : 

    datetime.datetime(year=df.year, month=df.month, day=df.day))

    store_id_relation = pd.read_csv('../input/store_id_relation.csv')

    date_info = pd.read_csv('../input/date_info.csv',parse_dates=['calendar_date']).rename(columns={'calendar_date':'visit_date'})

    air_reserve_by_date = air_reserve.groupby(['air_store_id','visit_date']).reserve_visitors.sum().reset_index(drop=False)

    hpg_reserve_by_date = hpg_reserve.groupby(['hpg_store_id','visit_date']).reserve_visitors.sum().reset_index(drop=False)

    hpg_store_info.drop(['latitude','longitude'],inplace=True,axis=1)

    test = air_visit_data.merge(air_store_info,on='air_store_id')

    test = test.merge(air_reserve_by_date,on=['air_store_id','visit_date'],how='left')

    test = test.merge(store_id_relation,on='air_store_id',how='left')

    test = test.merge(hpg_store_info,on='hpg_store_id',how='left')

    test = test.merge(hpg_reserve_by_date,on=['hpg_store_id','visit_date'],how='left')

    test = test.merge(date_info,on='visit_date',how='left')

    test['year'] = test.visit_date.dt.year

    test['month'] = test.visit_date.dt.month

    test.reserve_visitors_x = test.reserve_visitors_x.fillna(0)

    test.reserve_visitors_y = test.reserve_visitors_y.fillna(0)

    test.reserve_visitors_x = np.log1p(test.reserve_visitors_x)

    test.reserve_visitors_y = np.log1p(test.reserve_visitors_y)

    test = test.fillna(-1)

    test = test.sort_values(by='visit_date')

    test.visitors = np.log1p(test.visitors)

    return test[list(['id'])+list(columns)]
train = MungeTrain()

test = MungeTest(train.columns)
train.head()
test.head()
twoweeks = train.visit_date.max()-pd.Timedelta(days=14)
vistrain = train[train.visit_date<twoweeks].copy()

blindtrain = train[train.visit_date>=twoweeks].copy()

print(vistrain.shape)

print(blindtrain.shape)
features = ['day_of_week', 'holiday_flg', 'year']

for c in features:

    print(c)

    test[c+'_Count_Store'] = np.nan

    test[c+'_Mean_Store'] = np.nan

    test[c+'_Median_Store'] = np.nan

    test[c+'_Max_Store'] = np.nan

    test[c+'_Min_Store'] = np.nan

    test[c+'_Std_Store'] = np.nan

    

    vistrain[c+'_Count_Store'] = np.nan

    vistrain[c+'_Mean_Store'] = np.nan

    vistrain[c+'_Median_Store'] = np.nan

    vistrain[c+'_Max_Store'] = np.nan

    vistrain[c+'_Min_Store'] = np.nan

    vistrain[c+'_Std_Store'] = np.nan

    

    blindtrain[c+'_Count_Store'] = np.nan

    blindtrain[c+'_Mean_Store'] = np.nan

    blindtrain[c+'_Median_Store'] = np.nan

    blindtrain[c+'_Max_Store'] = np.nan

    blindtrain[c+'_Min_Store'] = np.nan

    blindtrain[c+'_Std_Store'] = np.nan

    

    test[[c+'_Count_Store',c+'_Mean_Store',

          c+'_Median_Store',c+'_Max_Store',

          c+'_Min_Store', c+'_Std_Store']] =  LeaveOneOut(vistrain,

                                                          test,

                                                          list(['air_store_id']),

                                                          c,

                                                          useLOO=True,

                                                          cut=0).values

    

    blindtrain[[c+'_Count_Store',c+'_Mean_Store',

                c+'_Median_Store',c+'_Max_Store',

                c+'_Min_Store', c+'_Std_Store']] =  LeaveOneOut(vistrain,

                                                                blindtrain,

                                                                list(['air_store_id']),

                                                                c,

                                                                useLOO=True,

                                                                cut=0).values



    vistrain[[c+'_Count_Store',c+'_Mean_Store',

              c+'_Median_Store',c+'_Max_Store',

              c+'_Min_Store', c+'_Std_Store']] =  LeaveOneOut(vistrain,

                                                              vistrain,

                                                              list(['air_store_id']),

                                                              c,

                                                              useLOO=True,

                                                              cut=1,

                                                              addNoise=False).values

features = ['air_store_id', 'air_genre_name',

            'air_area_name', 

            'hpg_store_id', 'hpg_genre_name', 'hpg_area_name',

            'day_of_week', 'holiday_flg', 'year', 'month']



for c in features:

    print(c)

    test[c+'_Count'] = np.nan

    test[c+'_Mean'] = np.nan

    test[c+'_Median'] = np.nan

    test[c+'_Max'] = np.nan

    test[c+'_Min'] = np.nan

    test[c+'_Std'] = np.nan

    

    vistrain[c+'_Count'] = np.nan

    vistrain[c+'_Mean'] = np.nan

    vistrain[c+'_Median'] = np.nan

    vistrain[c+'_Max'] = np.nan

    vistrain[c+'_Min'] = np.nan

    vistrain[c+'_Std'] = np.nan

    

    blindtrain[c+'_Count'] = np.nan

    blindtrain[c+'_Mean'] = np.nan

    blindtrain[c+'_Median'] = np.nan

    blindtrain[c+'_Max'] = np.nan

    blindtrain[c+'_Min'] = np.nan

    blindtrain[c+'_Std'] = np.nan

    

    test[[c+'_Count',c+'_Mean',

          c+'_Median',c+'_Max',

          c+'_Min', c+'_Std']] =  LeaveOneOut(vistrain.copy(),

                                              test.copy(),

                                              list([]),

                                              c,

                                              useLOO=False,

                                              cut=0,

                                              addNoise=False).values



    blindtrain[[c+'_Count',c+'_Mean',

                c+'_Median',c+'_Max',

                c+'_Min', c+'_Std']] =  LeaveOneOut(vistrain.copy(),

                                                    blindtrain.copy(),

                                                    list([]),

                                                    c,

                                                    useLOO=False,

                                                    cut=0,

                                                    addNoise=False).values





    vistrain[[c+'_Count',c+'_Mean',

              c+'_Median',c+'_Max',

              c+'_Min', c+'_Std']] =  LeaveOneOut(vistrain.copy(),

                                                  vistrain.copy(),

                                                  list([]),

                                                  c,

                                                  useLOO=True,

                                                  cut=1,

                                                  addNoise=False).values

    test.drop(c,inplace=True,axis=1)

    blindtrain.drop(c,inplace=True,axis=1)

    vistrain.drop(c,inplace=True,axis=1)

test = test.fillna(-1)

blindtrain = blindtrain.fillna(-1)

vistrain = vistrain.fillna(-1)
loofeatures = list(vistrain.columns[2:])
def GP1(data):

    v = pd.DataFrame()

    v["1"] = 0.600000*np.tanh(((((((3.0) - (data["day_of_week_Std_Store"]))) * (((data["day_of_week_Mean_Store"]) - (2.571430))))) - (0.425287)))

    v["2"] = 0.600000*np.tanh((((0.095238) + (np.minimum( ((((data["holiday_flg_Mean_Store"]) + (-3.0)))),  ((((data["day_of_week_Max_Store"]) - ((((7.0)) / 2.0))))))))/2.0))

    v["3"] = 0.571869*np.tanh(((data["holiday_flg_Min_Store"]) * (np.minimum( ((data["holiday_flg_Min_Store"])),  ((((((((3.750000) < (data["day_of_week_Mean_Store"]))*1.)) > (np.tanh((data["day_of_week_Mean_Store"]))))*1.)))))))

    v["4"] = 0.600000*np.tanh(np.minimum( ((((((data["day_of_week_Max_Store"]) + (0.202899))) + (-3.0)))),  (((((((((data["day_of_week_Max_Store"]) > (data["holiday_flg_Count_Store"]))*1.)) / 2.0)) / 2.0)))))

    v["5"] = 0.600000*np.tanh(((((((12.02259731292724609)) * ((((data["day_of_week_Median"]) > (data["air_store_id_Count"]))*1.)))) + ((-1.0*(((((((data["day_of_week_Median_Store"]) < (1.844440))*1.)) / 2.0))))))/2.0))

    v["6"] = 0.537058*np.tanh(np.maximum( ((np.minimum( ((((0.095238) * (data["holiday_flg_Median_Store"])))),  (((((2.571430) > (data["air_store_id_Mean"]))*1.)))))),  (((((data["holiday_flg_Median_Store"]) < (0.391304))*1.)))))

    v["7"] = 0.600000*np.tanh((((0.391304) + (np.minimum( ((0.322581)),  (((((((data["day_of_week_Std_Store"]) > (data["day_of_week_Mean_Store"]))*1.)) - (np.tanh((np.tanh((data["day_of_week_Std_Store"])))))))))))/2.0))

    v["8"] = 0.600000*np.tanh(np.tanh(((-1.0*(((((data["day_of_week_Median_Store"]) > (np.maximum( ((data["day_of_week_Max_Store"])),  ((3.0)))))*1.)))))))

    v["9"] = 0.600000*np.tanh((-1.0*((np.minimum( ((0.629032)),  ((np.minimum( (((((data["year_Max_Store"]) < (2.571430))*1.))),  (((((data["year_Median_Store"]) < (((0.884615) * 2.0)))*1.)))))))))))

    v["10"] = 0.600000*np.tanh((((((data["hpg_genre_name_Std"]) * (data["air_store_id_Std"]))) > ((((data["air_store_id_Count"]) > (((data["air_store_id_Median"]) * (data["air_store_id_Std"]))))*1.)))*1.))

    v["11"] = 0.561672*np.tanh(np.tanh((np.tanh((((np.tanh(((-1.0*(((((data["day_of_week_Max_Store"]) < ((((((data["day_of_week_Min_Store"]) / 2.0)) + (1.844440))/2.0)))*1.))))))) * 2.0))))))

    v["12"] = 0.600000*np.tanh(np.minimum( ((np.minimum( ((0.202899)),  ((((data["air_store_id_Max"]) - (data["day_of_week_Max_Store"]))))))),  ((np.minimum( ((0.095238)),  (((((data["day_of_week_Max_Store"]) < (2.571430))*1.))))))))

    v["13"] = 0.600000*np.tanh(np.minimum( ((((0.095238) / 2.0))),  ((((((-3.0) / 2.0)) + (((((((data["day_of_week_Median"]) > (data["day_of_week_Median_Store"]))*1.)) + (data["day_of_week_Median"]))/2.0)))))))

    v["14"] = 0.519359*np.tanh((((np.minimum( (((((-2.0) + (((data["year_Median_Store"]) - (0.322581))))/2.0))),  ((0.046154)))) + ((((data["reserve_visitors_x"]) > (data["year_Median_Store"]))*1.)))/2.0))

    v["15"] = 0.514671*np.tanh(((0.322581) * ((((data["holiday_flg_Mean_Store"]) < (((0.803030) * ((((((data["day_of_week_Max_Store"]) < (3.0))*1.)) * 2.0)))))*1.))))

    v["16"] = 0.541395*np.tanh(((((((0.803030) * (np.minimum( ((3.750000)),  (((((2.571430) < (data["air_store_id_Min"]))*1.))))))) / 2.0)) * (0.322581)))

    v["17"] = 0.580895*np.tanh(((-1.0) + (np.tanh((((0.202899) + (np.maximum( ((np.maximum( (((-1.0*((data["day_of_week_Median_Store"]))))),  ((data["day_of_week_Median_Store"]))))),  ((data["holiday_flg_Min_Store"]))))))))))

    v["18"] = 0.600000*np.tanh(((data["holiday_flg_Min_Store"]) * ((((((((data["day_of_week_Mean_Store"]) < (2.571430))*1.)) - ((((3.750000) < (data["day_of_week_Mean_Store"]))*1.)))) * (0.095238)))))

    v["19"] = 0.460988*np.tanh(((0.202899) * (((((((((data["year_Median_Store"]) - (3.750000))) * (data["day_of_week_Std_Store"]))) + (0.202899))) * (data["day_of_week_Std_Store"])))))

    v["20"] = 0.600000*np.tanh(((((data["air_store_id_Std"]) + (np.tanh(((-1.0*((data["air_store_id_Std"])))))))) * (np.tanh(((((data["day_of_week_Median_Store"]) > (data["air_store_id_Mean"]))*1.))))))

    v["21"] = 0.600000*np.tanh((((((data["air_store_id_Median"]) > (((3.750000) + (np.tanh((-1.0))))))*1.)) * (((0.629032) - (((0.322581) * 2.0))))))

    v["22"] = 0.600000*np.tanh(((0.095238) * (((data["day_of_week_Max_Store"]) * (((((((2.571430) + ((((2.571430) + (data["day_of_week_Median_Store"]))/2.0)))/2.0)) > (data["day_of_week_Max_Store"]))*1.))))))

    v["23"] = 0.577027*np.tanh(np.tanh((np.maximum( ((((data["air_store_id_Median"]) - (data["air_store_id_Median"])))),  (((((data["month_Count"]) < (((data["air_store_id_Median"]) * (2.0))))*1.)))))))

    v["24"] = 0.600000*np.tanh((((-1.0*((0.046154)))) * ((((data["air_store_id_Min"]) > (0.803030))*1.))))

    v["25"] = 0.567533*np.tanh(((0.095238) * (((((-3.0) * ((((data["day_of_week_Median_Store"]) < (1.440000))*1.)))) * ((((data["day_of_week_Median_Store"]) < (data["day_of_week_Mean_Store"]))*1.))))))

    v["26"] = 0.558273*np.tanh(((((((np.maximum( ((((2.073170) - (data["air_store_id_Mean"])))),  ((0.0)))) / 2.0)) / 2.0)) * (data["air_store_id_Median"])))

    v["27"] = 0.576910*np.tanh(((data["year_Min_Store"]) * (((1.294120) * (((data["year_Min_Store"]) * (((0.046154) * ((((data["day_of_week_Mean_Store"]) < ((2.77249646186828613)))*1.))))))))))

    v["28"] = 0.600000*np.tanh(((((np.minimum( ((0.046154)),  (((((data["year_Mean_Store"]) > (1.440000))*1.))))) * (1.440000))) * (((data["year_Mean_Store"]) - (3.0)))))

    v["29"] = 0.592030*np.tanh(np.minimum( ((0.0)),  ((np.tanh((((((-1.0*(((((data["holiday_flg_Count_Store"]) < (data["holiday_flg_Mean_Store"]))*1.))))) + (np.tanh((((data["holiday_flg_Count_Store"]) / 2.0)))))/2.0)))))))

    v["30"] = 0.546318*np.tanh(np.minimum( (((((data["day_of_week_Min_Store"]) < (((((((data["day_of_week_Median_Store"]) + (-3.0))/2.0)) > (0.425287))*1.)))*1.))),  (((((-1.0) + (data["day_of_week_Max_Store"]))/2.0)))))

    v["31"] = 0.493456*np.tanh(((0.095238) * (((((((((((3.0) > (data["air_store_id_Mean"]))*1.)) - (-2.0))) > (data["day_of_week_Median_Store"]))*1.)) - (0.629032)))))

    v["32"] = 0.600000*np.tanh(np.minimum( ((((data["day_of_week_Max_Store"]) * (((((((0.046154) * (((data["year_Max_Store"]) / 2.0)))) / 2.0)) - (0.046154)))))),  ((0.046154))))

    v["33"] = 0.600000*np.tanh(np.minimum( (((((2.571430) < (data["day_of_week_Min_Store"]))*1.))),  ((np.minimum( ((0.095238)),  ((np.minimum( (((((4.0) < (data["day_of_week_Max_Store"]))*1.))),  ((2.571430))))))))))

    v["34"] = 0.600000*np.tanh((((((((((data["holiday_flg_Max_Store"]) < (3.0))*1.)) + (data["reserve_visitors_y"]))) + ((((data["day_of_week_Min_Store"]) < (data["holiday_flg_Min_Store"]))*1.)))) * (0.046154)))

    v["35"] = 0.600000*np.tanh((-1.0*((((0.202899) * (((np.maximum( (((((data["day_of_week_Mean_Store"]) > (3.750000))*1.))),  (((((0.095238) > (data["day_of_week_Median_Store"]))*1.))))) / 2.0)))))))

    v["36"] = 0.600000*np.tanh(((0.046154) * (((((((((data["day_of_week_Mean_Store"]) - (1.844440))) > (((0.803030) * 2.0)))*1.)) > (0.095238))*1.))))

    v["37"] = 0.600000*np.tanh((-1.0*(((((((((data["air_store_id_Mean"]) > (3.0))*1.)) / 2.0)) * (((((((data["air_store_id_Count"]) < (data["air_store_id_Mean"]))*1.)) + (0.046154))/2.0)))))))

    v["38"] = 0.527330*np.tanh(((0.095238) * ((((((-1.0*((3.0)))) * (0.095238))) * (((data["air_store_id_Median"]) + ((-1.0*((3.0))))))))))

    v["39"] = 0.600000*np.tanh(np.minimum( (((((4.0) < (data["day_of_week_Median_Store"]))*1.))),  (((((0.202899) + ((((data["day_of_week_Mean_Store"]) < ((((1.440000) < (0.803030))*1.)))*1.)))/2.0)))))

    v["40"] = 0.449736*np.tanh(np.tanh(((-1.0*((((((((0.046154) > (1.0))*1.)) + ((((data["year_Max_Store"]) < (data["day_of_week_Mean_Store"]))*1.)))/2.0)))))))

    v["41"] = 0.600000*np.tanh(((np.tanh(((((data["day_of_week_Min_Store"]) > ((3.0)))*1.)))) * ((((4.0) < ((((data["day_of_week_Mean_Store"]) + ((3.0)))/2.0)))*1.))))

    v["42"] = 0.600000*np.tanh(((((((0.046154) * (data["day_of_week_Std_Store"]))) * (data["day_of_week_Std_Store"]))) * ((((-1.0) + (((data["reserve_visitors_x"]) * (data["reserve_visitors_x"]))))/2.0))))

    v["43"] = 0.600000*np.tanh(np.minimum( ((data["day_of_week_Count_Store"])),  ((np.minimum( ((0.046154)),  (((((((((0.046154) * 2.0)) + ((7.0)))) < (((data["day_of_week_Mean_Store"]) * 2.0)))*1.))))))))

    v["44"] = 0.590623*np.tanh(np.minimum( ((0.322581)),  (((((0.99218511581420898)) - ((((0.0) < (data["day_of_week_Std_Store"]))*1.)))))))

    v["45"] = 0.600000*np.tanh((((((data["day_of_week_Median_Store"]) + (0.322581))) > (((data["day_of_week_Mean_Store"]) + (1.844440))))*1.))

    v["46"] = 0.600000*np.tanh((((0.0) < (-3.0))*1.))

    v["47"] = 0.424067*np.tanh(np.minimum( ((np.minimum( ((0.322581)),  (((((data["day_of_week_Mean_Store"]) < (((data["holiday_flg_Min_Store"]) + (0.322581))))*1.)))))),  (((((1.0) < (data["day_of_week_Mean_Store"]))*1.)))))

    v["48"] = 0.600000*np.tanh(((data["air_store_id_Min"]) * (((data["air_store_id_Min"]) * ((-1.0*((((np.tanh((3.0))) - (np.tanh((2.571430))))))))))))

    v["49"] = 0.574565*np.tanh(np.minimum( (((((2.571430) < (data["day_of_week_Min_Store"]))*1.))),  (((((((data["holiday_flg_Median_Store"]) < (data["day_of_week_Min_Store"]))*1.)) * ((((data["day_of_week_Median_Store"]) < (data["holiday_flg_Median_Store"]))*1.)))))))

    v["50"] = 0.600000*np.tanh(((0.095238) * (np.minimum( (((((((data["holiday_flg_Count_Store"]) > (data["day_of_week_Mean_Store"]))*1.)) + (-1.0)))),  ((0.411765))))))

    v["51"] = 0.452080*np.tanh(np.minimum( (((((data["day_of_week_Mean_Store"]) > (2.073170))*1.))),  ((((0.629032) * (((data["year_Median_Store"]) - (data["air_store_id_Mean"]))))))))

    v["52"] = 0.578199*np.tanh((((((((data["year_Mean_Store"]) < (((((((data["year_Mean_Store"]) < (1.440000))*1.)) + (data["air_store_id_Median"]))/2.0)))*1.)) * (0.391304))) * (data["year_Mean_Store"])))

    v["53"] = 0.600000*np.tanh((-1.0*((np.minimum( ((3.750000)),  (((((((((0.046154) + ((((((3.750000) < (data["day_of_week_Mean_Store"]))*1.)) / 2.0)))/2.0)) / 2.0)) / 2.0))))))))

    v["54"] = 0.600000*np.tanh((-1.0*(((((((((((((0.261905) * (data["holiday_flg_Mean_Store"]))) < (data["reserve_visitors_x"]))*1.)) / 2.0)) / 2.0)) * (0.202899))))))

    v["55"] = 0.600000*np.tanh((((((np.tanh(((((4.0) < (data["day_of_week_Mean_Store"]))*1.)))) + ((((data["day_of_week_Mean_Store"]) > ((((7.0)) / 2.0)))*1.)))/2.0)) * (0.095238)))

    v["56"] = 0.544677*np.tanh(((0.046154) * ((((data["air_store_id_Mean"]) < (((1.844440) + ((((data["year_Median_Store"]) < (((2.073170) + (0.322581))))*1.)))))*1.))))

    v["57"] = 0.579019*np.tanh(((0.411765) - (0.425287)))

    v["58"] = 0.600000*np.tanh(np.minimum( (((((((data["year_Count_Store"]) < (3.0))*1.)) * ((-1.0*((0.202899))))))),  ((((data["year_Median_Store"]) - (np.tanh((1.294120))))))))

    v["59"] = 0.542801*np.tanh(((((3.0) * (data["day_of_week_Mean"]))) * (((data["month_Std"]) - ((((0.629032) + (np.tanh((((1.440000) * 2.0)))))/2.0))))))

    v["60"] = 0.600000*np.tanh(((((((((-1.0*((0.397849)))) + ((((0.261905) < (((data["day_of_week_Median"]) * (0.095238))))*1.)))/2.0)) * (0.095238))) / 2.0))

    v["61"] = 0.600000*np.tanh((-1.0*((((((((((((data["day_of_week_Mean_Store"]) > (2.073170))*1.)) < ((((data["day_of_week_Median_Store"]) > (data["day_of_week_Mean_Store"]))*1.)))*1.)) * (0.095238))) / 2.0)))))

    v["62"] = 0.600000*np.tanh(((np.tanh((((0.046154) - (np.tanh((((data["air_store_id_Median"]) + ((-1.0*((data["day_of_week_Median_Store"])))))))))))) * (0.046154)))

    v["63"] = 0.600000*np.tanh(((0.046154) * ((((((((1.844440) + ((((3.0) > (data["air_store_id_Mean"]))*1.)))) > (data["day_of_week_Mean_Store"]))*1.)) / 2.0))))

    v["64"] = 0.600000*np.tanh((-1.0*((((0.046154) * (((np.minimum( ((0.095238)),  ((np.tanh((data["day_of_week_Median_Store"])))))) * (data["day_of_week_Min_Store"]))))))))

    v["65"] = 0.600000*np.tanh((((((data["year_Count_Store"]) < ((((((((0.425287) > (data["day_of_week_Std_Store"]))*1.)) + (0.425287))) * 2.0)))*1.)) * (((data["day_of_week_Std_Store"]) * 2.0))))

    v["66"] = 0.541160*np.tanh(((-1.0) + (np.tanh(((((((-1.0) + ((((2.073170) < (data["year_Min_Store"]))*1.)))) + (data["day_of_week_Max"]))/2.0))))))

    v["67"] = 0.564837*np.tanh((((np.minimum( ((0.202899)),  (((((0.803030) < (data["air_area_name_Min"]))*1.))))) + (((0.046154) * (0.261905))))/2.0))

    v["68"] = 0.442821*np.tanh(((data["air_store_id_Median"]) * (((data["air_store_id_Median"]) * (((-1.0) + ((((np.tanh((data["holiday_flg_Max_Store"]))) + (np.tanh((data["year_Max_Store"]))))/2.0))))))))

    v["69"] = 0.472358*np.tanh(np.maximum( ((0.0)),  (((((((np.tanh((data["day_of_week_Median_Store"]))) > (0.884615))*1.)) - (np.tanh((data["air_store_id_Median"]))))))))

    v["70"] = 0.600000*np.tanh(((((np.tanh((np.maximum( ((3.0)),  ((data["year_Max_Store"])))))) + (-1.0))) * (data["air_store_id_Count"])))

    v["71"] = 0.541512*np.tanh((-1.0*(((((data["day_of_week_Std_Store"]) > ((((4.0) + (0.397849))/2.0)))*1.)))))

    v["72"] = 0.600000*np.tanh(((((((-1.0*((0.046154)))) * (((0.046154) / 2.0)))) + (((((0.046154) / 2.0)) / 2.0)))/2.0))

    v["73"] = 0.600000*np.tanh(((((0.095238) * ((((data["day_of_week_Count_Store"]) > (0.095238))*1.)))) * ((((2.571430) > (((0.095238) + (data["day_of_week_Max_Store"]))))*1.))))

    v["74"] = 0.600000*np.tanh(np.minimum( ((0.046154)),  (((0.0)))))

    v["75"] = 0.600000*np.tanh(np.tanh((((((((((0.202899) > (((data["holiday_flg_Mean_Store"]) + (data["day_of_week_Count_Store"]))))*1.)) > ((((0.202899) + (data["day_of_week_Count_Store"]))/2.0)))*1.)) / 2.0))))

    v["76"] = 0.480094*np.tanh(np.minimum( ((((data["air_store_id_Median"]) * (((np.tanh((data["air_store_id_Count"]))) + (-1.0)))))),  ((((np.tanh((data["day_of_week_Mean_Store"]))) + (data["day_of_week_Max_Store"]))))))

    v["77"] = 0.568588*np.tanh((-1.0*((np.minimum( ((np.minimum( (((((data["day_of_week_Std_Store"]) < (data["day_of_week_Median_Store"]))*1.))),  ((0.095238))))),  (((((data["day_of_week_Median_Store"]) < (1.440000))*1.))))))))

    v["78"] = 0.600000*np.tanh((((((((data["day_of_week_Std_Store"]) > ((((((data["day_of_week_Count_Store"]) + (0.046154))/2.0)) - (0.046154))))*1.)) * (data["day_of_week_Count_Store"]))) * (0.046154)))

    v["79"] = 0.600000*np.tanh(((1.0) + (np.tanh(((-1.0*((np.maximum( ((data["air_store_id_Median"])),  (((((data["air_store_id_Count"]) + ((((data["air_store_id_Count"]) > (data["day_of_week_Max"]))*1.)))/2.0))))))))))))

    v["80"] = 0.345419*np.tanh(((0.202899) * (((0.046154) * (((((((0.397849) < (0.411765))*1.)) + ((-1.0*((data["air_area_name_Mean"])))))/2.0))))))

    v["81"] = 0.600000*np.tanh(((0.095238) * ((((2.0) < (((((data["air_store_id_Median"]) + (np.tanh((-1.0))))) / 2.0)))*1.))))

    v["82"] = 0.453839*np.tanh(((0.095238) * ((((2.571430) > (data["holiday_flg_Max_Store"]))*1.))))

    v["83"] = 0.600000*np.tanh((((((((-1.0) + (np.tanh((2.073170))))/2.0)) * (data["air_store_id_Count"]))) * ((((data["year_Max_Store"]) > (data["air_store_id_Count"]))*1.))))

    v["84"] = 0.600000*np.tanh((((-3.0) > (4.0))*1.))

    v["85"] = 0.600000*np.tanh((((((1.844440) < (data["day_of_week_Std_Store"]))*1.)) * (((((((1.844440) < (0.095238))*1.)) + ((((data["day_of_week_Mean_Store"]) < (1.844440))*1.)))/2.0))))

    v["86"] = 0.449971*np.tanh(((np.minimum( ((0.202899)),  (((((0.629032) > (data["day_of_week_Median_Store"]))*1.))))) * ((((data["day_of_week_Mean_Store"]) > (0.046154))*1.))))

    v["87"] = 0.600000*np.tanh((((-3.0) > ((0.0)))*1.))

    v["88"] = 0.572807*np.tanh(np.minimum( ((0.046154)),  (((((data["year_Min_Store"]) > (((((((((data["day_of_week_Mean_Store"]) > (3.750000))*1.)) * (1.844440))) + (data["day_of_week_Max_Store"]))/2.0)))*1.)))))

    v["89"] = 0.600000*np.tanh(((((((((((data["day_of_week_Std_Store"]) - (0.095238))) > (0.202899))*1.)) + (-1.0))/2.0)) * (0.095238)))

    v["90"] = 0.600000*np.tanh((-1.0*((((np.minimum( ((0.046154)),  ((((((((0.425287) + (4.0))/2.0)) > (data["day_of_week_Mean_Store"]))*1.))))) / 2.0)))))

    v["91"] = 0.600000*np.tanh((((((((data["air_store_id_Mean"]) / 2.0)) > (0.803030))*1.)) * ((((1.0) + ((-1.0*((np.tanh((data["air_store_id_Median"])))))))/2.0))))

    v["92"] = 0.600000*np.tanh(((((1.81595969200134277)) < (1.294120))*1.))

    v["93"] = 0.494862*np.tanh(np.minimum( ((((((((data["year_Max_Store"]) + (0.046154))/2.0)) > (data["day_of_week_Mean_Store"]))*1.))),  (((((0.261905) > ((((data["year_Max_Store"]) + (-2.0))/2.0)))*1.)))))

    v["94"] = 0.565892*np.tanh(((data["holiday_flg_Count_Store"]) * (((0.046154) * ((-1.0*((np.tanh((np.tanh((np.tanh(((((2.571430) > (data["holiday_flg_Count_Store"]))*1.)))))))))))))))

    v["95"] = 0.488416*np.tanh(((((((((((0.046154) / 2.0)) * (0.425287))) * (((0.046154) / 2.0)))) * (data["holiday_flg_Max_Store"]))) * (data["holiday_flg_Max_Store"])))

    v["96"] = 0.600000*np.tanh(np.minimum( ((np.minimum( ((0.202899)),  (((((3.750000) < (data["day_of_week_Min_Store"]))*1.)))))),  (((((np.minimum( ((data["holiday_flg_Mean_Store"])),  ((data["day_of_week_Median_Store"])))) < (data["day_of_week_Min_Store"]))*1.)))))

    v["97"] = 0.320688*np.tanh((((3.750000) < (((data["air_store_id_Min"]) - (data["holiday_flg_Count_Store"]))))*1.))

    v["98"] = 0.449385*np.tanh((((np.tanh((((((((np.tanh((-1.0))) + (((-1.0) - (data["air_store_id_Min"]))))/2.0)) + (data["day_of_week_Max"]))/2.0)))) + (-1.0))/2.0))

    v["99"] = 0.600000*np.tanh((((((data["day_of_week_Min_Store"]) < ((((((data["reserve_visitors_y"]) + (np.minimum( ((0.095238)),  ((data["reserve_visitors_y"])))))) + (data["day_of_week_Std_Store"]))/2.0)))*1.)) * (0.095238)))

    v["100"] = 0.577847*np.tanh(((np.tanh((data["holiday_flg_Mean"]))) - (np.tanh((((0.777778) - (((-2.0) * (((0.322581) * (data["air_area_name_Mean"])))))))))))

    return v.sum(axis=1)+2.771291



def GP2(data):

    v = pd.DataFrame()

    v["1"] = 0.600000*np.tanh((((((((((data["day_of_week_Median_Store"]) - (3.0))) * 2.0)) * 2.0)) + (np.tanh((data["day_of_week_Min_Store"]))))/2.0))

    v["2"] = 0.600000*np.tanh((((((3.750000) < (data["day_of_week_Mean_Store"]))*1.)) + (((0.095238) + ((((np.minimum( ((data["year_Median_Store"])),  ((data["day_of_week_Mean_Store"])))) + (-3.0))/2.0))))))

    v["3"] = 0.600000*np.tanh(np.minimum( ((((0.095238) * (data["holiday_flg_Min_Store"])))),  ((((np.minimum( ((((0.095238) + (data["day_of_week_Max_Store"])))),  ((data["year_Max_Store"])))) - (data["holiday_flg_Median"]))))))

    v["4"] = 0.600000*np.tanh((((((data["day_of_week_Count"]) * (((data["air_store_id_Median"]) * ((((2.073170) < (((data["air_store_id_Median"]) / 2.0)))*1.)))))) + ((-1.0*((0.046154)))))/2.0))

    v["5"] = 0.600000*np.tanh(((((np.tanh((0.322581))) * ((((data["day_of_week_Count"]) > (data["holiday_flg_Count"]))*1.)))) / 2.0))

    v["6"] = 0.550654*np.tanh(np.tanh((((((data["holiday_flg_Mean_Store"]) + ((-1.0*((data["air_store_id_Mean"])))))) - (np.minimum( (((((data["day_of_week_Median"]) < (data["air_store_id_Mean"]))*1.))),  ((0.046154))))))))

    v["7"] = 0.600000*np.tanh(((data["holiday_flg_Min_Store"]) * (np.minimum( (((((data["holiday_flg_Min_Store"]) > (1.0))*1.))),  ((((((((data["day_of_week_Std_Store"]) > (data["day_of_week_Count_Store"]))*1.)) > (data["day_of_week_Count_Store"]))*1.)))))))

    v["8"] = 0.600000*np.tanh(((((0.425287) * ((-1.0*((((((((3.0) < (data["holiday_flg_Median_Store"]))*1.)) + ((-1.0*((0.425287)))))/2.0))))))) / 2.0))

    v["9"] = 0.583239*np.tanh(((data["hpg_genre_name_Std"]) * (np.tanh((((((((data["hpg_genre_name_Std"]) > (0.411765))*1.)) < (data["hpg_genre_name_Std"]))*1.))))))

    v["10"] = 0.600000*np.tanh((-1.0*((((((((((2.0) + ((((data["day_of_week_Std_Store"]) + ((((data["day_of_week_Median"]) < (data["day_of_week_Count_Store"]))*1.)))/2.0)))/2.0)) > (data["day_of_week_Max_Store"]))*1.)) * 2.0)))))

    v["11"] = 0.600000*np.tanh((((((data["day_of_week_Median"]) < (data["year_Mean_Store"]))*1.)) * (((((data["day_of_week_Median"]) - (((1.440000) * 2.0)))) / 2.0))))

    v["12"] = 0.519828*np.tanh(np.tanh(((((((0.202899) + ((((0.095238) + ((-1.0*((data["day_of_week_Std_Store"])))))/2.0)))/2.0)) + (((((0.01152277272194624)) > (data["day_of_week_Std_Store"]))*1.))))))

    v["13"] = 0.600000*np.tanh((((((((((data["air_store_id_Std"]) > (np.maximum( ((data["day_of_week_Count_Store"])),  ((0.884615)))))*1.)) * 2.0)) * 2.0)) * 2.0))

    v["14"] = 0.600000*np.tanh(((np.minimum( ((((((data["holiday_flg_Std_Store"]) - (((data["holiday_flg_Min_Store"]) / 2.0)))) * 2.0))),  ((((data["day_of_week_Count_Store"]) - (2.073170)))))) * (0.046154)))

    v["15"] = 0.570229*np.tanh((((np.minimum( ((((data["year_Median_Store"]) / 2.0))),  ((1.0)))) + (np.tanh((((data["year_Median_Store"]) - ((((1.0) + (data["month_Count"]))/2.0)))))))/2.0))

    v["16"] = 0.600000*np.tanh((((((((data["air_store_id_Median"]) < (2.571430))*1.)) * (((((((data["holiday_flg_Median_Store"]) < (data["air_store_id_Median"]))*1.)) + ((-1.0*((0.261905)))))/2.0)))) / 2.0))

    v["17"] = 0.556867*np.tanh(((0.397849) * (((data["day_of_week_Median_Store"]) * (((data["day_of_week_Median_Store"]) * ((((((data["year_Min_Store"]) - (0.095238))) > (data["day_of_week_Median_Store"]))*1.))))))))

    v["18"] = 0.600000*np.tanh((-1.0*(((((((0.046154) * (data["holiday_flg_Min_Store"]))) + ((((((((((data["day_of_week_Mean_Store"]) / 2.0)) / 2.0)) > (data["day_of_week_Count_Store"]))*1.)) / 2.0)))/2.0)))))

    v["19"] = 0.600000*np.tanh(((((0.202899) * ((((data["reserve_visitors_x"]) + ((((1.844440) < ((((data["year_Mean_Store"]) + (0.202899))/2.0)))*1.)))/2.0)))) * (data["day_of_week_Std_Store"])))

    v["20"] = 0.562844*np.tanh(((np.minimum( ((data["day_of_week_Std_Store"])),  (((((data["day_of_week_Std_Store"]) > (0.777778))*1.))))) * (((data["day_of_week_Median_Store"]) * ((-1.0*((np.tanh((0.046154))))))))))

    v["21"] = 0.506232*np.tanh(((((0.629032) * (data["day_of_week_Median_Store"]))) * (((((0.046154) * (data["day_of_week_Median_Store"]))) * ((((data["day_of_week_Median_Store"]) < (2.571430))*1.))))))

    v["22"] = 0.490643*np.tanh(((((data["holiday_flg_Max_Store"]) / 2.0)) * (((0.046154) * (((((-2.0) * 2.0)) + ((((data["year_Mean_Store"]) + (data["year_Max_Store"]))/2.0))))))))

    v["23"] = 0.532838*np.tanh(np.tanh((np.tanh((np.tanh((((((data["day_of_week_Mean_Store"]) - ((((data["day_of_week_Mean_Store"]) + (data["day_of_week_Median_Store"]))/2.0)))) / 2.0))))))))

    v["24"] = 0.600000*np.tanh(np.minimum( (((((data["day_of_week_Count_Store"]) > (((0.884615) * (3.0))))*1.))),  ((((0.046154) / 2.0)))))

    v["25"] = 0.465677*np.tanh(((np.tanh((np.minimum( ((((data["year_Mean_Store"]) - (data["air_store_id_Median"])))),  ((((data["air_store_id_Median"]) - ((((data["year_Mean_Store"]) > (data["month_Std"]))*1.))))))))) / 2.0))

    v["26"] = 0.564837*np.tanh(((np.maximum( ((data["air_store_id_Std"])),  ((data["air_store_id_Min"])))) * (((np.minimum( ((0.046154)),  (((((0.777778) > (data["air_store_id_Min"]))*1.))))) * (1.440000)))))

    v["27"] = 0.487595*np.tanh((((-1.0*((np.tanh(((((((((((0.803030) > (data["day_of_week_Max_Store"]))*1.)) + (((data["holiday_flg_Max_Store"]) / 2.0)))/2.0)) > (data["day_of_week_Max_Store"]))*1.))))))) / 2.0))

    v["28"] = 0.544560*np.tanh(((data["day_of_week_Std_Store"]) * (((((data["day_of_week_Std"]) / 2.0)) - (((np.maximum( ((data["day_of_week_Std"])),  (((((0.803030) < (data["day_of_week_Std"]))*1.))))) / 2.0))))))

    v["29"] = 0.600000*np.tanh((((((0.777778) < (data["air_store_id_Std"]))*1.)) * ((((((data["holiday_flg_Std_Store"]) < (data["air_store_id_Std"]))*1.)) * (0.095238)))))

    v["30"] = 0.574448*np.tanh(((0.202899) * (np.maximum( (((((2.0) > (data["year_Count_Store"]))*1.))),  ((((((0.046154) * (data["holiday_flg_Max_Store"]))) * (0.202899))))))))

    v["31"] = 0.542801*np.tanh(((data["year_Count_Store"]) * (np.maximum( ((((0.397849) - (data["air_area_name_Std"])))),  ((0.0))))))

    v["32"] = 0.487712*np.tanh(((data["year_Count_Store"]) * (((data["year_Count_Store"]) * (((np.tanh((data["year_Count_Store"]))) + ((-1.0*((np.tanh(((5.0)))))))))))))

    v["33"] = 0.556398*np.tanh(((((((1.0) * (((1.0) - (np.tanh((data["air_store_id_Count"]))))))) * ((10.0)))) * (1.0)))

    v["34"] = 0.600000*np.tanh(((1.0) - (np.maximum( (((((np.tanh((((0.777778) - ((-1.0*((1.0)))))))) > (data["year_Std_Store"]))*1.))),  ((data["year_Std_Store"]))))))

    v["35"] = 0.600000*np.tanh((-1.0*((np.tanh((((((2.571430) - (data["reserve_visitors_x"]))) * (np.minimum( ((0.095238)),  ((data["reserve_visitors_x"])))))))))))

    v["36"] = 0.534948*np.tanh(((np.minimum( (((((np.tanh((0.884615))) + ((-1.0*((data["hpg_genre_name_Min"])))))/2.0))),  (((((data["month_Std"]) + ((-1.0*((data["air_area_name_Std"])))))/2.0))))) / 2.0))

    v["37"] = 0.586169*np.tanh(((((data["year_Min_Store"]) * (0.046154))) * (np.maximum( ((((2.0) * ((((data["reserve_visitors_y"]) > (2.0))*1.))))),  ((0.095238))))))

    v["38"] = 0.600000*np.tanh((((((data["holiday_flg_Min_Store"]) > ((((-1.0) + (data["month_Max"]))/2.0)))*1.)) * ((((((-1.0*((0.202899)))) / 2.0)) / 2.0))))

    v["39"] = 0.600000*np.tanh(np.minimum( ((((((((3.750000) < (data["year_Mean_Store"]))*1.)) > ((-1.0*(((((0.629032) > (data["hpg_area_name_Std"]))*1.))))))*1.))),  ((0.046154))))

    v["40"] = 0.600000*np.tanh((-1.0*((((((data["year_Count_Store"]) * (((((0.261905) * ((((data["year_Count_Store"]) < (((1.440000) * 2.0)))*1.)))) / 2.0)))) / 2.0)))))

    v["41"] = 0.600000*np.tanh((((data["hpg_store_id_Std"]) < (0.095238))*1.))

    v["42"] = 0.426646*np.tanh((-1.0*(((((-1.0) + (np.tanh((((((((((data["holiday_flg_Count"]) + (0.391304))/2.0)) + (-3.0))/2.0)) * 2.0)))))/2.0)))))

    v["43"] = 0.591326*np.tanh((((((1.0) < (data["hpg_store_id_Std"]))*1.)) * (((np.tanh((0.261905))) / 2.0))))

    v["44"] = 0.600000*np.tanh((((((((-1.0*((((0.046154) * ((((1.440000) < (data["hpg_store_id_Min"]))*1.))))))) * (data["hpg_store_id_Min"]))) * (1.440000))) / 2.0))

    v["45"] = 0.599766*np.tanh((((((((0.629032) > (data["year_Std_Store"]))*1.)) * (((0.629032) * ((((data["day_of_week_Std_Store"]) < (0.095238))*1.)))))) / 2.0))

    v["46"] = 0.600000*np.tanh(((np.tanh((np.maximum( ((((data["month_Max"]) - (data["air_store_id_Max"])))),  ((((data["air_store_id_Max"]) - (1.294120)))))))) - (np.tanh((data["hpg_area_name_Mean"])))))

    v["47"] = 0.600000*np.tanh((((0.884615) < (0.411765))*1.))

    v["48"] = 0.523579*np.tanh((((-1.0) + (np.tanh(((((((data["day_of_week_Median_Store"]) > ((1.43529331684112549)))*1.)) + (data["day_of_week_Median_Store"]))))))/2.0))

    v["49"] = 0.600000*np.tanh(((0.046154) * ((((((data["day_of_week_Mean_Store"]) < ((((((2.571430) * 2.0)) + (0.322581))/2.0)))*1.)) * (((data["day_of_week_Mean_Store"]) / 2.0))))))

    v["50"] = 0.515491*np.tanh((-1.0*(((((((np.tanh((data["year_Std_Store"]))) > (0.425287))*1.)) * (((0.046154) * ((((3.0) > (data["year_Mean_Store"]))*1.)))))))))

    v["51"] = 0.600000*np.tanh(np.minimum( ((np.minimum( ((0.046154)),  (((((2.0) < (data["day_of_week_Median_Store"]))*1.)))))),  (((((data["day_of_week_Median_Store"]) < (data["air_area_name_Mean"]))*1.)))))

    v["52"] = 0.513616*np.tanh(((data["day_of_week_Std_Store"]) * ((-1.0*((((np.minimum( ((0.095238)),  (((((((1.844440) / 2.0)) > (data["day_of_week_Std_Store"]))*1.))))) * (data["day_of_week_Std_Store"]))))))))

    v["53"] = 0.600000*np.tanh(np.minimum( ((0.425287)),  ((((((((0.202899) + (data["day_of_week_Std"]))/2.0)) > (data["day_of_week_Median_Store"]))*1.)))))

    v["54"] = 0.600000*np.tanh(np.minimum( (((((data["air_store_id_Std"]) < ((((np.tanh((np.tanh((3.0))))) < (data["air_store_id_Std"]))*1.)))*1.))),  ((np.tanh((0.046154))))))

    v["55"] = 0.600000*np.tanh(((-3.0) * ((0.0))))

    v["56"] = 0.510686*np.tanh((((data["day_of_week_Std_Store"]) > (data["year_Count_Store"]))*1.))

    v["57"] = 0.600000*np.tanh(((np.tanh((-3.0))) + (np.tanh((((((data["year_Count_Store"]) + (-1.0))) + ((((data["year_Count_Store"]) < (data["air_area_name_Mean"]))*1.))))))))

    v["58"] = 0.600000*np.tanh(((0.095238) * ((((((data["holiday_flg_Max_Store"]) + (0.202899))) < (3.0))*1.))))

    v["59"] = 0.591561*np.tanh(((((data["holiday_flg_Max_Store"]) / 2.0)) * (((((np.tanh((data["holiday_flg_Max_Store"]))) - (1.0))) * (data["holiday_flg_Max_Store"])))))

    v["60"] = 0.599766*np.tanh((((((((((0.261905) * ((((data["day_of_week_Median_Store"]) < (2.571430))*1.)))) + ((((data["day_of_week_Median_Store"]) > ((4.0)))*1.)))/2.0)) / 2.0)) / 2.0))

    v["61"] = 0.600000*np.tanh(((0.046154) * ((((data["holiday_flg_Median_Store"]) + ((-1.0*((((data["air_area_name_Mean"]) * (((data["air_area_name_Mean"]) * (0.411765)))))))))/2.0))))

    v["62"] = 0.600000*np.tanh(((0.202899) * ((((((data["air_store_id_Mean"]) / 2.0)) < ((((data["day_of_week_Std_Store"]) < (0.425287))*1.)))*1.))))

    v["63"] = 0.600000*np.tanh(((((data["day_of_week_Std"]) - (((1.294120) / 2.0)))) * (((((data["day_of_week_Std"]) - (0.046154))) - (0.803030)))))

    v["64"] = 0.600000*np.tanh(((0.095238) * ((-1.0*(((((data["holiday_flg_Max_Store"]) < (2.0))*1.)))))))

    v["65"] = 0.159289*np.tanh((-1.0*((((data["month_Std"]) * (((0.397849) - (((0.202899) * 2.0)))))))))

    v["66"] = 0.600000*np.tanh(((np.minimum( ((((data["hpg_area_name_Std"]) - (0.411765)))),  ((0.0)))) * (data["month_Max"])))

    v["67"] = 0.600000*np.tanh(((0.322581) * (((0.884615) - (np.maximum( ((data["hpg_area_name_Std"])),  ((0.884615))))))))

    v["68"] = 0.534831*np.tanh(((((np.tanh(((((data["day_of_week_Std_Store"]) < (data["air_area_name_Min"]))*1.)))) * (0.095238))) * (np.tanh(((((data["hpg_area_name_Std"]) < (data["air_area_name_Min"]))*1.))))))

    v["69"] = 0.562610*np.tanh(((-2.0) + (((np.tanh(((((data["hpg_genre_name_Max"]) + (0.425287))/2.0)))) + (np.tanh(((((0.046154) + (data["hpg_genre_name_Max"]))/2.0))))))))

    v["70"] = 0.590623*np.tanh(((data["month_Std"]) - (np.maximum( (((((3.750000) < (data["air_store_id_Min"]))*1.))),  ((0.803030))))))

    v["71"] = 0.600000*np.tanh((((-1.0) + (np.tanh(((((((((3.750000) < (data["holiday_flg_Max_Store"]))*1.)) * (data["holiday_flg_Max_Store"]))) - ((-1.0*((1.844440)))))))))/2.0))

    v["72"] = 0.600000*np.tanh(((0.046154) * ((((((((data["air_store_id_Median"]) > (data["day_of_week_Median_Store"]))*1.)) * ((((3.0) > (data["day_of_week_Median_Store"]))*1.)))) - (0.425287)))))

    v["73"] = 0.593085*np.tanh(np.tanh((np.minimum( ((((((data["day_of_week_Min_Store"]) / 2.0)) * (((data["day_of_week_Min_Store"]) * ((((data["day_of_week_Mean_Store"]) < (data["day_of_week_Min_Store"]))*1.))))))),  ((0.391304))))))

    v["74"] = 0.600000*np.tanh((((((data["day_of_week_Median_Store"]) < (1.844440))*1.)) * (((data["day_of_week_Std_Store"]) * (((0.095238) * ((-1.0*((1.294120))))))))))

    v["75"] = 0.600000*np.tanh((((((((np.tanh(((((data["air_store_id_Mean"]) < (data["year_Max_Store"]))*1.)))) > ((((data["year_Max_Store"]) + (np.tanh((-2.0))))/2.0)))*1.)) / 2.0)) / 2.0))

    v["76"] = 0.600000*np.tanh(((((0.046154) * (data["day_of_week_Count_Store"]))) * (((((0.046154) * (data["day_of_week_Count_Store"]))) * (((0.046154) * (data["day_of_week_Count_Store"])))))))

    v["77"] = 0.600000*np.tanh(np.maximum( ((0.0)),  ((-3.0))))

    v["78"] = 0.600000*np.tanh((((-1.0) > (-1.0))*1.))

    v["79"] = 0.591444*np.tanh(((data["day_of_week_Mean"]) * (((data["day_of_week_Mean"]) * ((((((np.tanh((data["hpg_store_id_Count"]))) + (-1.0))/2.0)) * (data["day_of_week_Mean"])))))))

    v["80"] = 0.600000*np.tanh(((((8.80587291717529297)) < (2.073170))*1.))

    v["81"] = 0.600000*np.tanh(((((8.0)) < (0.0))*1.))

    v["82"] = 0.600000*np.tanh((((4.0) < (0.322581))*1.))

    v["83"] = 0.600000*np.tanh((((((((-1.0) + (np.tanh((data["month_Max"]))))/2.0)) * (data["month_Count"]))) * (((data["month_Count"]) + (-3.0)))))

    v["84"] = 0.395351*np.tanh(((0.046154) * (0.046154)))

    v["85"] = 0.600000*np.tanh((((((((((-1.0) + (0.095238))/2.0)) > (0.0))*1.)) + ((((-1.0) + (np.tanh((data["day_of_week_Mean"]))))/2.0)))/2.0))

    v["86"] = 0.599883*np.tanh(((np.maximum( ((np.tanh((((data["day_of_week_Mean"]) - (np.tanh((data["day_of_week_Count_Store"])))))))),  ((np.tanh((data["day_of_week_Count_Store"])))))) + (np.tanh((-3.0)))))

    v["87"] = 0.453839*np.tanh(((((((((data["day_of_week_Max_Store"]) > (((2.571430) * 2.0)))*1.)) / 2.0)) + (((((((((2.50835967063903809)) > (data["day_of_week_Max_Store"]))*1.)) / 2.0)) / 2.0)))/2.0))

    v["88"] = 0.574800*np.tanh(np.minimum( ((np.minimum( ((0.0)),  ((((data["day_of_week_Max_Store"]) - (data["month_Std"]))))))),  ((((((data["day_of_week_Mean_Store"]) - (((data["day_of_week_Max_Store"]) / 2.0)))) / 2.0)))))

    v["89"] = 0.592616*np.tanh(((np.minimum( (((((-1.0) + (data["day_of_week_Max_Store"]))/2.0))),  (((((data["day_of_week_Max_Store"]) < (np.maximum( ((2.073170)),  ((((0.803030) * 2.0))))))*1.))))) / 2.0))

    v["90"] = 0.600000*np.tanh(((((np.tanh((-3.0))) + (np.tanh((data["day_of_week_Median"]))))) * (((np.tanh((np.tanh((-3.0))))) + (data["day_of_week_Max_Store"])))))

    v["91"] = 0.597773*np.tanh(((data["year_Mean_Store"]) * ((((0.777778) > ((((data["day_of_week_Count_Store"]) + (data["year_Mean_Store"]))/2.0)))*1.))))

    v["92"] = 0.357374*np.tanh((((((((-1.0*(((((((data["year_Median_Store"]) > (data["day_of_week_Max_Store"]))*1.)) / 2.0))))) * (data["day_of_week_Max_Store"]))) * (0.046154))) * (data["day_of_week_Max_Store"])))

    v["93"] = 0.600000*np.tanh(np.minimum( ((((data["year_Median_Store"]) - (data["month_Std"])))),  ((((((((((data["month_Std"]) - (0.803030))) / 2.0)) / 2.0)) * (data["day_of_week_Max_Store"]))))))

    v["94"] = 0.599883*np.tanh(np.minimum( ((0.046154)),  (((((3.750000) < (((((((1.844440) / 2.0)) * (0.095238))) + (data["year_Median_Store"]))))*1.)))))

    v["95"] = 0.581012*np.tanh(((((0.046154) * (((((data["day_of_week_Median_Store"]) * ((((-1.0*((0.397849)))) / 2.0)))) / 2.0)))) / 2.0))

    v["96"] = 0.600000*np.tanh(((((0.046154) * (1.440000))) * ((((data["air_store_id_Std"]) < ((((np.tanh((1.440000))) < (data["air_store_id_Std"]))*1.)))*1.))))

    v["97"] = 0.599883*np.tanh(np.tanh((np.tanh((np.tanh(((((data["day_of_week_Max_Store"]) > (((data["hpg_genre_name_Count"]) + (0.397849))))*1.))))))))

    v["98"] = 0.406368*np.tanh((((-1.0) + ((((np.tanh((data["hpg_genre_name_Count"]))) + ((((data["hpg_genre_name_Max"]) < (data["hpg_genre_name_Count"]))*1.)))/2.0)))/2.0))

    v["99"] = 0.600000*np.tanh(np.minimum( (((((data["year_Mean_Store"]) > (data["hpg_store_id_Median"]))*1.))),  ((((0.095238) * ((((((data["hpg_store_id_Median"]) > (data["day_of_week_Median_Store"]))*1.)) - (0.046154))))))))

    v["100"] = 0.599883*np.tanh(((0.046154) * ((((((0.046154) * (0.095238))) + ((-1.0*((0.202899)))))/2.0))))

    return v.sum(axis=1)+2.771291

    

def GP3(data):

    v = pd.DataFrame()

    v["1"] = 0.600000*np.tanh(((-3.0) + (((((-2.0) + (((((-2.0) + (data["year_Median_Store"]))) + (data["day_of_week_Min_Store"]))))) + (data["holiday_flg_Mean_Store"])))))

    v["2"] = 0.598594*np.tanh(((np.minimum( (((((data["day_of_week_Median_Store"]) + (-3.0))/2.0))),  ((((data["air_store_id_Median"]) - (2.571430)))))) + (((data["day_of_week_Median_Store"]) - (data["air_store_id_Mean"])))))

    v["3"] = 0.600000*np.tanh(np.minimum( ((((0.046154) * (((data["day_of_week_Count_Store"]) - (2.073170)))))),  ((((data["day_of_week_Max_Store"]) - ((((2.571430) + (data["holiday_flg_Mean"]))/2.0)))))))

    v["4"] = 0.562610*np.tanh(((((data["holiday_flg_Std"]) + ((((3.750000) < (np.maximum( ((data["air_store_id_Mean"])),  ((data["day_of_week_Min_Store"])))))*1.)))) + (np.tanh(((-1.0*((data["day_of_week_Min_Store"]))))))))

    v["5"] = 0.504825*np.tanh(np.minimum( ((0.046154)),  ((((-1.0) + (((((data["day_of_week_Max_Store"]) - ((((((data["day_of_week_Max_Store"]) > (data["day_of_week_Count_Store"]))*1.)) * 2.0)))) / 2.0)))))))

    v["6"] = 0.600000*np.tanh(((np.minimum( ((data["holiday_flg_Count_Store"])),  ((0.803030)))) * (((np.minimum( ((0.391304)),  ((data["reserve_visitors_x"])))) * (((data["reserve_visitors_x"]) - (1.844440)))))))

    v["7"] = 0.580660*np.tanh(((((4.0) * (0.046154))) * (((data["day_of_week_Mean"]) - (((((data["holiday_flg_Count"]) / 2.0)) / 2.0))))))

    v["8"] = 0.600000*np.tanh((((0.046154) + ((((((((((((data["hpg_store_id_Count"]) < (3.750000))*1.)) * 2.0)) * 2.0)) * 2.0)) * 2.0)))/2.0))

    v["9"] = 0.600000*np.tanh(((((((data["month_Std"]) - (0.803030))) + (((((((data["month_Std"]) - (0.411765))) * 2.0)) - (0.803030))))) * 2.0))

    v["10"] = 0.600000*np.tanh(((0.202899) * (((((((((data["year_Std_Store"]) - (0.046154))) > (0.884615))*1.)) > (((data["year_Std_Store"]) * (0.884615))))*1.))))

    v["11"] = 0.599766*np.tanh(((((-1.0*((0.411765)))) + (((0.397849) + (((np.tanh(((((data["air_store_id_Count"]) < (3.0))*1.)))) * 2.0)))))/2.0))

    v["12"] = 0.600000*np.tanh((-1.0*((((((((data["year_Max_Store"]) < (data["air_store_id_Median"]))*1.)) < ((((3.0)) + (((data["air_store_id_Median"]) - (((data["year_Max_Store"]) * 2.0)))))))*1.)))))

    v["13"] = 0.600000*np.tanh((((((1.294120) > (((data["air_store_id_Mean"]) / 2.0)))*1.)) * (((((0.046154) * (data["air_store_id_Median"]))) * (((data["air_store_id_Mean"]) / 2.0))))))

    v["14"] = 0.511272*np.tanh(((data["day_of_week_Mean_Store"]) * (np.minimum( ((((-1.0) + (((data["day_of_week_Mean_Store"]) / 2.0))))),  ((((-1.0) + (np.tanh((data["day_of_week_Max_Store"]))))))))))

    v["15"] = 0.588045*np.tanh(np.minimum( (((((data["air_store_id_Mean"]) < (0.884615))*1.))),  ((np.minimum( (((((1.440000) > (np.tanh((data["air_store_id_Count"]))))*1.))),  ((np.tanh((2.571430)))))))))

    v["16"] = 0.600000*np.tanh(((((((data["air_store_id_Median"]) - (0.046154))) - (data["air_store_id_Mean"]))) * ((-1.0*((0.425287))))))

    v["17"] = 0.600000*np.tanh(((((((np.tanh((np.tanh((((data["month_Max"]) - (data["air_store_id_Count"]))))))) / 2.0)) / 2.0)) / 2.0))

    v["18"] = 0.600000*np.tanh(np.minimum( ((((np.tanh((((data["day_of_week_Std_Store"]) - (0.629032))))) * ((-1.0*((0.261905))))))),  ((((0.046154) * (data["day_of_week_Std_Store"]))))))

    v["19"] = 0.421020*np.tanh((((((data["day_of_week_Min_Store"]) + (-1.0))/2.0)) * ((((-1.0) + (np.tanh(((((data["day_of_week_Min_Store"]) + (0.0))/2.0)))))/2.0))))

    v["20"] = 0.600000*np.tanh(((np.tanh((((data["year_Median_Store"]) - (0.095238))))) - (np.tanh((((2.571430) - ((((data["year_Median_Store"]) < (2.571430))*1.))))))))

    v["21"] = 0.600000*np.tanh(np.tanh((np.tanh((np.tanh((np.tanh((np.tanh((((np.tanh((data["year_Median_Store"]))) - (np.minimum( ((1.0)),  ((data["year_Median_Store"]))))))))))))))))

    v["22"] = 0.599414*np.tanh(np.minimum( ((np.minimum( ((0.411765)),  (((((2.073170) < (data["air_area_name_Mean"]))*1.)))))),  ((np.minimum( ((1.844440)),  (((((data["year_Max_Store"]) < (2.073170))*1.))))))))

    v["23"] = 0.600000*np.tanh(((np.minimum( ((3.0)),  ((data["year_Min_Store"])))) * (((np.minimum( ((3.0)),  ((data["year_Min_Store"])))) * (((data["month_Std"]) - (0.803030)))))))

    v["24"] = 0.538699*np.tanh(np.minimum( ((0.322581)),  (((((1.0) < (((data["hpg_store_id_Std"]) + (np.minimum( (((((1.294120) < (data["reserve_visitors_y"]))*1.))),  ((0.322581)))))))*1.)))))

    v["25"] = 0.582184*np.tanh(((0.411765) - (0.425287)))

    v["26"] = 0.541512*np.tanh(((1.440000) * (((0.046154) * ((((9.0)) + ((-1.0*((((data["month_Count"]) - (0.202899))))))))))))

    v["27"] = 0.551709*np.tanh(((0.046154) * (np.tanh((((data["hpg_store_id_Mean"]) * ((((((data["air_store_id_Mean"]) < (3.0))*1.)) - (0.391304)))))))))

    v["28"] = 0.569057*np.tanh(np.minimum( ((((0.095238) * (((data["holiday_flg_Max_Store"]) - (4.0)))))),  ((((0.046154) * ((((data["year_Min_Store"]) > (2.0))*1.)))))))

    v["29"] = 0.490174*np.tanh(((((0.202899) * (data["day_of_week_Std_Store"]))) * (np.maximum( ((((((5.64322614669799805)) > (data["holiday_flg_Max"]))*1.))),  (((((data["air_area_name_Median"]) < (2.571430))*1.)))))))

    v["30"] = 0.600000*np.tanh((((((data["year_Mean_Store"]) > ((((np.tanh((1.294120))) + (2.0))/2.0)))*1.)) * ((((-1.0) + (np.tanh((data["year_Mean_Store"]))))/2.0))))

    v["31"] = 0.600000*np.tanh((-1.0*(((((((data["hpg_store_id_Min"]) > (2.571430))*1.)) * (np.tanh((np.minimum( ((np.tanh((0.397849)))),  ((data["hpg_store_id_Min"])))))))))))

    v["32"] = 0.600000*np.tanh((((-3.0) > (0.0))*1.))

    v["33"] = 0.600000*np.tanh(((np.minimum( (((((data["day_of_week_Std_Store"]) > (((0.411765) * (2.073170))))*1.))),  ((0.046154)))) * (-2.0)))

    v["34"] = 0.600000*np.tanh(((((((((data["year_Max_Store"]) < ((2.95706820487976074)))*1.)) + (0.046154))/2.0)) * ((((data["day_of_week_Count_Store"]) < (data["year_Max_Store"]))*1.))))

    v["35"] = 0.532838*np.tanh((-1.0*((((((((((data["air_store_id_Mean"]) > (3.750000))*1.)) / 2.0)) + (np.minimum( (((((data["air_store_id_Mean"]) > (data["hpg_store_id_Mean"]))*1.))),  ((0.046154)))))/2.0)))))

    v["36"] = 0.600000*np.tanh(((data["day_of_week_Std_Store"]) * (((data["year_Median_Store"]) * (((data["day_of_week_Std_Store"]) * (((-1.0) + (np.tanh((data["year_Mean_Store"])))))))))))

    v["37"] = 0.600000*np.tanh(((0.046154) * (((data["holiday_flg_Min_Store"]) * ((((data["year_Max_Store"]) > (((((((data["year_Max_Store"]) + (0.095238))/2.0)) + ((6.20778131484985352)))/2.0)))*1.))))))

    v["38"] = 0.523344*np.tanh((((((data["holiday_flg_Median_Store"]) > (data["day_of_week_Median_Store"]))*1.)) * (((data["day_of_week_Mean"]) * (((0.046154) * ((((data["hpg_store_id_Median"]) > (data["holiday_flg_Median_Store"]))*1.))))))))

    v["39"] = 0.571752*np.tanh(((data["day_of_week_Std_Store"]) * ((((-1.0*((((((data["hpg_store_id_Mean"]) * (((0.046154) * (data["hpg_store_id_Mean"]))))) * (0.391304)))))) / 2.0))))

    v["40"] = 0.600000*np.tanh(((0.425287) * ((((((np.maximum( ((data["year_Mean_Store"])),  ((3.750000)))) < (data["day_of_week_Mean_Store"]))*1.)) / 2.0))))

    v["41"] = 0.600000*np.tanh(((0.202899) * ((-1.0*((((np.minimum( ((0.629032)),  ((((data["hpg_store_id_Median"]) - (data["day_of_week_Mean"])))))) / 2.0)))))))

    v["42"] = 0.557101*np.tanh((((((data["day_of_week_Median_Store"]) < ((((((data["day_of_week_Min_Store"]) - (((0.629032) / 2.0)))) + (data["day_of_week_Min_Store"]))/2.0)))*1.)) / 2.0))

    v["43"] = 0.433678*np.tanh((((((((((data["air_store_id_Mean"]) / 2.0)) / 2.0)) < (0.777778))*1.)) * (((((((-1.0*((data["air_store_id_Min"])))) + (0.777778))/2.0)) / 2.0))))

    v["44"] = 0.600000*np.tanh((-1.0*(((((((data["air_area_name_Min"]) < ((((data["day_of_week_Min_Store"]) + ((((data["hpg_area_name_Std"]) > (data["day_of_week_Min_Store"]))*1.)))/2.0)))*1.)) * (0.046154))))))

    v["45"] = 0.600000*np.tanh(((0.046154) * (((np.maximum( ((data["year_Min_Store"])),  (((((data["year_Mean_Store"]) + ((-1.0*((0.411765)))))/2.0))))) - (((data["day_of_week_Min_Store"]) / 2.0))))))

    v["46"] = 0.600000*np.tanh(((data["air_store_id_Median"]) * ((((((data["air_store_id_Median"]) - (0.777778))) > (((data["holiday_flg_Mean_Store"]) * 2.0)))*1.))))

    v["47"] = 0.572573*np.tanh((((((((((((1.440000) > (data["day_of_week_Min_Store"]))*1.)) * (0.095238))) * (data["day_of_week_Min_Store"]))) * (data["day_of_week_Min_Store"]))) / 2.0))

    v["48"] = 0.600000*np.tanh(np.tanh((np.tanh((np.tanh((np.tanh(((((np.tanh(((((0.397849) > (data["hpg_area_name_Std"]))*1.)))) + ((-1.0*((0.046154)))))/2.0))))))))))

    v["49"] = 0.600000*np.tanh((((((((((data["day_of_week_Mean_Store"]) - (0.046154))) < (np.minimum( ((data["day_of_week_Std_Store"])),  ((1.844440)))))*1.)) * (data["day_of_week_Std_Store"]))) * (data["year_Max_Store"])))

    v["50"] = 0.600000*np.tanh(np.minimum( ((((data["day_of_week_Max_Store"]) - ((((2.073170) > (data["day_of_week_Max_Store"]))*1.))))),  ((np.minimum( (((((2.073170) > (data["day_of_week_Max_Store"]))*1.))),  ((0.411765)))))))

    v["51"] = 0.598945*np.tanh(((((-1.0) * (data["day_of_week_Median_Store"]))) * (((data["day_of_week_Count_Store"]) * ((((data["day_of_week_Max_Store"]) < (data["day_of_week_Median_Store"]))*1.))))))

    v["52"] = 0.600000*np.tanh((((((((((data["day_of_week_Median_Store"]) / 2.0)) > (data["year_Median_Store"]))*1.)) - (np.tanh(((((data["year_Max_Store"]) < (data["day_of_week_Median_Store"]))*1.)))))) / 2.0))

    v["53"] = 0.554874*np.tanh(((((0.322581) / 2.0)) * (np.minimum( (((((data["day_of_week_Max_Store"]) < (2.571430))*1.))),  (((((data["year_Mean_Store"]) < (3.0))*1.)))))))

    v["54"] = 0.600000*np.tanh((((((data["holiday_flg_Mean_Store"]) > (3.0))*1.)) * ((-1.0*(((((((data["day_of_week_Mean_Store"]) < ((((0.884615) + (data["year_Median_Store"]))/2.0)))*1.)) / 2.0)))))))

    v["55"] = 0.524282*np.tanh(np.tanh(((-1.0*((((0.095238) * ((((2.073170) > ((((((data["day_of_week_Mean_Store"]) + (data["year_Mean_Store"]))/2.0)) - (0.095238))))*1.)))))))))

    v["56"] = 0.600000*np.tanh(((np.minimum( (((((1.440000) > (data["day_of_week_Min_Store"]))*1.))),  (((((data["air_store_id_Mean"]) < (data["day_of_week_Median"]))*1.))))) * (((0.046154) * (data["day_of_week_Min_Store"])))))

    v["57"] = 0.600000*np.tanh((-1.0*((((((np.tanh((data["year_Mean_Store"]))) * ((((data["holiday_flg_Max_Store"]) < (np.maximum( ((data["day_of_week_Max_Store"])),  ((data["holiday_flg_Median_Store"])))))*1.)))) * (0.095238))))))

    v["58"] = 0.600000*np.tanh(((((((((data["year_Median_Store"]) * 2.0)) < (data["holiday_flg_Median_Store"]))*1.)) > ((((((data["year_Median_Store"]) * 2.0)) < (data["holiday_flg_Mean_Store"]))*1.)))*1.))

    v["59"] = 0.600000*np.tanh(((np.minimum( (((((data["holiday_flg_Mean_Store"]) < (0.391304))*1.))),  ((0.425287)))) * 2.0))

    v["60"] = 0.600000*np.tanh(((0.095238) * (((((((((5.0)) - (0.095238))) - (data["year_Mean_Store"]))) < (1.440000))*1.))))

    v["61"] = 0.600000*np.tanh((((((((1.844440) < ((((data["air_store_id_Mean"]) + (data["air_store_id_Min"]))/2.0)))*1.)) * ((-1.0*((0.046154)))))) / 2.0))

    v["62"] = 0.600000*np.tanh((((0.803030) < (0.391304))*1.))

    v["63"] = 0.600000*np.tanh((((-1.0) + (np.tanh((np.maximum( ((1.0)),  ((((((np.tanh((data["day_of_week_Max_Store"]))) * (data["day_of_week_Max_Store"]))) * (data["day_of_week_Max_Store"])))))))))/2.0))

    v["64"] = 0.599648*np.tanh(((((((((np.tanh((data["day_of_week_Max"]))) - (np.tanh((data["holiday_flg_Max"]))))) * (data["holiday_flg_Max"]))) * (data["air_area_name_Count"]))) * ((12.25948238372802734))))

    v["65"] = 0.541864*np.tanh(((((0.425287) * (0.046154))) * (np.tanh((((3.750000) * (np.tanh((((data["year_Max_Store"]) - (3.750000)))))))))))

    v["66"] = 0.433327*np.tanh((-1.0*((((0.095238) * (((data["hpg_genre_name_Min"]) * (((4.0) * ((((data["air_store_id_Mean"]) > (3.750000))*1.)))))))))))

    v["67"] = 0.600000*np.tanh((((((np.tanh((0.411765))) > (data["holiday_flg_Std_Store"]))*1.)) * (((((0.425287) * (data["holiday_flg_Std_Store"]))) / 2.0))))

    v["68"] = 0.548896*np.tanh(((0.095238) * ((((((0.397849) + (data["air_store_id_Mean"]))) < (np.maximum( ((data["air_store_id_Median"])),  ((2.073170)))))*1.))))

    v["69"] = 0.540574*np.tanh((((data["day_of_week_Count_Store"]) < ((((data["air_store_id_Min"]) > ((((data["air_store_id_Median"]) + (((0.425287) - ((((data["air_store_id_Median"]) < (1.844440))*1.)))))/2.0)))*1.)))*1.))

    v["70"] = 0.575737*np.tanh((((((-1.0*((((data["hpg_genre_name_Min"]) * (((0.095238) * ((((data["air_store_id_Min"]) > (0.777778))*1.))))))))) * (0.777778))) / 2.0))

    v["71"] = 0.599648*np.tanh(((((((((((((2.073170) < (data["year_Mean_Store"]))*1.)) / 2.0)) * (0.046154))) + ((((4.0) < (data["year_Mean_Store"]))*1.)))/2.0)) / 2.0))

    v["72"] = 0.524634*np.tanh((((-1.0) + (np.tanh((((data["day_of_week_Median"]) + ((((((data["air_store_id_Median"]) < (3.0))*1.)) + (-1.0))))))))/2.0))

    v["73"] = 0.600000*np.tanh(((np.tanh((-3.0))) + ((((((((0.046154) > (((data["day_of_week_Std"]) - (np.tanh((data["year_Max_Store"]))))))*1.)) / 2.0)) * 2.0))))

    v["74"] = 0.541278*np.tanh(((((((np.tanh((data["year_Max_Store"]))) - (np.tanh((3.750000))))) * (np.minimum( ((3.750000)),  ((data["year_Mean_Store"])))))) * (data["year_Mean_Store"])))

    v["75"] = 0.576206*np.tanh(((0.261905) * (np.minimum( ((0.095238)),  (((((data["air_area_name_Count"]) > ((9.0)))*1.)))))))

    v["76"] = 0.554171*np.tanh(((((((-1.0*((0.202899)))) + (((0.884615) - (data["air_area_name_Std"]))))/2.0)) * ((((0.884615) < (data["air_area_name_Std"]))*1.))))

    v["77"] = 0.599883*np.tanh(((np.minimum( ((data["hpg_store_id_Median"])),  ((data["hpg_store_id_Mean"])))) * ((-1.0*((((0.046154) * ((((4.0) < (data["hpg_store_id_Median"]))*1.)))))))))

    v["78"] = 0.544560*np.tanh((((((((((1.0)) - (np.tanh((data["air_store_id_Median"]))))) * (np.tanh((data["air_store_id_Median"]))))) * (0.046154))) * (data["day_of_week_Max"])))

    v["79"] = 0.599766*np.tanh((((-1.0*((0.411765)))) * (np.tanh((np.tanh(((((data["day_of_week_Max"]) < (data["year_Max_Store"]))*1.))))))))

    v["80"] = 0.600000*np.tanh(np.minimum( ((0.0)),  ((((data["day_of_week_Max_Store"]) * (((np.tanh((data["day_of_week_Median"]))) + (np.tanh((-3.0))))))))))

    v["81"] = 0.433796*np.tanh((((((((0.046154) * (data["year_Mean_Store"]))) + ((((0.777778) < (data["air_area_name_Min"]))*1.)))/2.0)) * (((0.046154) * (data["year_Mean_Store"])))))

    v["82"] = 0.600000*np.tanh((((0.0) < (-3.0))*1.))

    v["83"] = 0.600000*np.tanh(np.minimum( (((((data["day_of_week_Max"]) < (((data["air_store_id_Count"]) - (((data["hpg_genre_name_Min"]) / 2.0)))))*1.))),  ((((data["day_of_week_Std"]) - (np.tanh((data["hpg_genre_name_Min"]))))))))

    v["84"] = 0.600000*np.tanh(((data["day_of_week_Max_Store"]) * (((0.046154) * (((data["day_of_week_Min_Store"]) * ((((data["day_of_week_Max_Store"]) < ((((2.571430) + (2.073170))/2.0)))*1.))))))))

    v["85"] = 0.600000*np.tanh(((0.095238) * ((((((3.0) < ((((3.750000) + (((data["day_of_week_Max_Store"]) / 2.0)))/2.0)))*1.)) / 2.0))))

    v["86"] = 0.479156*np.tanh(((-1.0) + (np.tanh((np.maximum( ((0.397849)),  ((np.maximum( ((data["day_of_week_Min_Store"])),  ((((0.411765) * (data["day_of_week_Max"])))))))))))))

    v["87"] = 0.600000*np.tanh((-1.0*((((((((((0.777778) + (data["day_of_week_Min_Store"]))/2.0)) > (data["air_store_id_Mean"]))*1.)) * (data["day_of_week_Min_Store"]))))))

    v["88"] = 0.559445*np.tanh(((((np.tanh((data["air_store_id_Median"]))) * ((((-1.0) + ((((data["day_of_week_Median_Store"]) > (np.tanh((np.tanh((2.073170))))))*1.)))/2.0)))) / 2.0))

    v["89"] = 0.600000*np.tanh(((1.0) - (np.tanh((((data["air_store_id_Median"]) + ((((1.294120) > ((((data["air_store_id_Mean"]) + (0.391304))/2.0)))*1.))))))))

    v["90"] = 0.600000*np.tanh(((data["day_of_week_Median_Store"]) * (((data["day_of_week_Median_Store"]) * (np.minimum( ((data["day_of_week_Median_Store"])),  (((((np.tanh((data["day_of_week_Min_Store"]))) > (data["day_of_week_Median_Store"]))*1.)))))))))

    v["91"] = 0.600000*np.tanh(((((((((data["day_of_week_Min_Store"]) > (2.073170))*1.)) * ((-1.0*((0.046154)))))) + ((((((data["air_store_id_Min"]) > (data["day_of_week_Min_Store"]))*1.)) / 2.0)))/2.0))

    v["92"] = 0.600000*np.tanh(np.tanh((np.minimum( ((0.261905)),  (((((1.844440) < (data["day_of_week_Std_Store"]))*1.)))))))

    v["93"] = 0.584528*np.tanh(((np.minimum( ((((data["holiday_flg_Median_Store"]) * (((data["holiday_flg_Median_Store"]) * (0.046154)))))),  ((((data["year_Mean_Store"]) - (data["holiday_flg_Median_Store"])))))) * (0.095238)))

    v["94"] = 0.600000*np.tanh(((0.046154) * ((((((((((0.26585704088211060)) + (0.046154))/2.0)) > (((2.0) - (((data["day_of_week_Max_Store"]) / 2.0)))))*1.)) / 2.0))))

    v["95"] = 0.600000*np.tanh((((0.0) < (-1.0))*1.))

    v["96"] = 0.407658*np.tanh(((((0.095238) * ((((1.440000) > ((((0.046154) + (data["air_store_id_Mean"]))/2.0)))*1.)))) * (((0.095238) * (data["air_store_id_Mean"])))))

    v["97"] = 0.550420*np.tanh((-1.0*((((data["holiday_flg_Std_Store"]) * (((data["holiday_flg_Std_Store"]) * (((((0.095238) * ((((data["holiday_flg_Std_Store"]) < (0.803030))*1.)))) / 2.0)))))))))

    v["98"] = 0.458879*np.tanh(((((0.095238) + ((((((data["year_Min_Store"]) / 2.0)) > (data["day_of_week_Std_Store"]))*1.)))) * ((((np.tanh((2.0))) < (data["day_of_week_Std_Store"]))*1.))))

    v["99"] = 0.600000*np.tanh(((0.046154) * ((((0.425287) > (((data["hpg_genre_name_Std"]) * (((data["hpg_genre_name_Std"]) * (data["air_genre_name_Std"]))))))*1.))))

    v["100"] = 0.600000*np.tanh(((data["holiday_flg_Std_Store"]) * (np.tanh((((((((((data["day_of_week_Std_Store"]) < (0.095238))*1.)) * 2.0)) > (data["year_Min_Store"]))*1.))))))

    return v.sum(axis=1)+2.771291



def GP(data):

    return (GP1(data)+GP2(data)+GP3(data))/3.
print(np.sqrt(mean_squared_error(vistrain.visitors,GP(vistrain))))

print(np.sqrt(mean_squared_error(blindtrain.visitors,

                                 GP(blindtrain))))
test.visitors = np.expm1(GP(test))

test[['id','visitors']].to_csv('gpsubmission.csv.gz',index=False,float_format='%.6f',compression='gzip')