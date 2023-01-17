#IMPORTING REQUIRED LIBRARIES

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

import datetime



from lightgbm.sklearn import LGBMRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_log_error,mean_squared_error

from sklearn.model_selection import KFold

import lightgbm as lgb

import pickle



import gc

gc.enable()





import warnings

warnings.filterwarnings("ignore")

%matplotlib inline
#DATASET VIEW

path1="/kaggle/input/ashrae-energy-prediction/"

path="/kaggle/input/ashrae-eda-and-104-models/"

data_files=list(os.listdir(path1))

df_files=pd.DataFrame(data_files,columns=['File_Name'])

df_files['Size_in_MB']=df_files.File_Name.apply(lambda x:round(os.stat(path1+x).st_size/(1024*1024),2))



with pd.option_context('display.max_rows', None, 'display.max_columns', None):

    display(df_files.sort_values('File_Name'))
%%time

#READING TRAIN DATASET

print('READING TRAIN DATASET...')

df_train=pd.read_csv(path1+'train.csv')



print('READING WEATHER TRAIN DATASET...')

df_weather_train=pd.read_csv(path1+'weather_train.csv')



print('READING WEATHER TEST DATASET...')

df_weather_test=pd.read_csv(path1+'weather_test.csv')



print('READING BUILDING METADATA...')

df_building_metadata=pd.read_csv(path1+'building_metadata.csv')



print('DATA READING COMPLETE')
#All FUNCTIONS



#FUNCTION FOR PROVIDING FEATURE SUMMARY

def feature_summary(df_fa):

    print('DataFrame shape')

    print('rows:',df_fa.shape[0])

    print('cols:',df_fa.shape[1])

    col_list=['Null','Unique_Count','Data_type','Max/Min','Mean','Std','Skewness','Sample_values']

    df=pd.DataFrame(index=df_fa.columns,columns=col_list)

    df['Null']=list([len(df_fa[col][df_fa[col].isnull()]) for i,col in enumerate(df_fa.columns)])

    #df['%_Null']=list([len(df_fa[col][df_fa[col].isnull()])/df_fa.shape[0]*100 for i,col in enumerate(df_fa.columns)])

    df['Unique_Count']=list([len(df_fa[col].unique()) for i,col in enumerate(df_fa.columns)])

    df['Data_type']=list([df_fa[col].dtype for i,col in enumerate(df_fa.columns)])

    for i,col in enumerate(df_fa.columns):

        if 'float' in str(df_fa[col].dtype) or 'int' in str(df_fa[col].dtype):

            df.at[col,'Max/Min']=str(round(df_fa[col].max(),2))+'/'+str(round(df_fa[col].min(),2))

            df.at[col,'Mean']=df_fa[col].mean()

            df.at[col,'Std']=df_fa[col].std()

            df.at[col,'Skewness']=df_fa[col].skew()

        elif 'datetime64[ns]' in str(df_fa[col].dtype):

            df.at[col,'Max/Min']=str(df_fa[col].max())+'/'+str(df_fa[col].min())

        df.at[col,'Sample_values']=list(df_fa[col].unique())

    display(df_fa.head())       

    return(df.fillna('-'))



#FUNCTION FOR READING DICTIONARY ITEMS AND HANDLING KEYERROR

def get_val(x,col):

    try:

        y=x[col]

    except:

        y=np.nan

    return(y)



#FUNCTION FOR CALCULATING RMSE

def rmse(y,pred):

    return(mean_squared_error(y,pred)**0.5)
#CONVERTING timestamp TO DATATIME FIELD

df_train['timestamp']=pd.to_datetime(df_train['timestamp'])

#FEATURE SUMMARY FOR TRAIN DATASET

feature_summary(df_train)
#PLOT FOR METER READING BY DATES

plt.figure(figsize=(50,10))

plt.title("METER READING BY DATES",fontsize=40,color='b')

plt.xlabel("Dates",fontsize=40,color='b')

plt.ylabel("Meter Reading",fontsize=40,color='b')

plt.xticks(fontsize=35)

plt.yticks(fontsize=35)

plt.plot(df_train['timestamp'],df_train['meter_reading'],color='green',linewidth=3)



plt.show()
#PIE CHART CHECKING DISTRIBUTION OF TARGET FEATURE

pie_labels=['METER READING LESS THAN 1000 UNITS : '+str(df_train['meter_reading'][df_train.meter_reading<1000.0].count()),

            'METER READING GREATER AND EQUAL TO 1000 UNITS : '+str(df_train['meter_reading'][df_train.meter_reading>=1000.0].count())            

           ]

pie_share=[df_train['meter_reading'][df_train.meter_reading<1000.0].count()/df_train['meter_reading'].count(),

           df_train['meter_reading'][df_train.meter_reading>=1000.0].count()/df_train['meter_reading'].count()

          ]

figureObject, axesObject = plt.subplots(figsize=(6,6))

pie_colors=('orange','grey')

pie_explode=(.15,.15)

axesObject.pie(pie_share,labels=pie_labels,explode=pie_explode,autopct='%.2f%%',colors=pie_colors,startangle=45)

axesObject.axis('equal')

plt.title('OBSERVATION % WITH METER READING LESS THAN 1000 UNITS AND OTHERWISE',color='blue',fontsize=12)

plt.show()
#PIE CHART CHECKING DISTRIBUTION OF TARGET FEATURE

pie_labels=['METER READING LESS THAN 5000 UNITS : '+str(df_train['meter_reading'][df_train.meter_reading<5000.0].count()),

            'METER READING GREATER AND EQUAL TO 5000 UNITS : '+str(df_train['meter_reading'][df_train.meter_reading>=5000.0].count())            

           ]

pie_share=[df_train['meter_reading'][df_train.meter_reading<5000.0].count()/df_train['meter_reading'].count(),

           df_train['meter_reading'][df_train.meter_reading>=5000.0].count()/df_train['meter_reading'].count()

          ]

figureObject, axesObject = plt.subplots(figsize=(6,6))

pie_colors=('orange','grey')

pie_explode=(.50,.25)

axesObject.pie(pie_share,labels=pie_labels,explode=pie_explode,autopct='%.2f%%',colors=pie_colors,startangle=45)

axesObject.axis('equal')

plt.title('OBSERVATION % WITH METER READING LESS THAN 5000 UNITS AND OTHERWISE',color='blue',fontsize=12)

plt.show()
#PIE CHART CHECKING DISTRIBUTION OF TARGET FEATURE

pie_labels=['METER READING LESS THAN 10,000 UNITS : '+str(df_train['meter_reading'][df_train.meter_reading<10000.0].count()),

            'METER READING GREATER AND EQUAL TO 10,000 UNITS : '+str(df_train['meter_reading'][df_train.meter_reading>=10000.0].count())            

           ]

pie_share=[df_train['meter_reading'][df_train.meter_reading<10000.0].count()/df_train['meter_reading'].count(),

           df_train['meter_reading'][df_train.meter_reading>=10000.0].count()/df_train['meter_reading'].count()

          ]

figureObject, axesObject = plt.subplots(figsize=(6,6))

pie_colors=('orange','grey')

pie_explode=(.50,.25)

axesObject.pie(pie_share,labels=pie_labels,explode=pie_explode,autopct='%.2f%%',colors=pie_colors,startangle=45)

axesObject.axis('equal')

plt.title('OBSERVATION % WITH METER READING LESS THAN 10,000 UNITS AND OTHERWISE',color='blue',fontsize=12)

plt.show()
#PIE CHART SHOWING DATA CATEGORIZATION BY METER TYPE

pie_labels=['METER TYPE 0 : '+str(df_train['meter'][df_train.meter==0].count()),

            'METER TYPE 1 : '+str(df_train['meter'][df_train.meter==1].count()),

            'METER TYPE 2 : '+str(df_train['meter'][df_train.meter==2].count()),

            'METER TYPE 3 : '+str(df_train['meter'][df_train.meter==3].count())

           ]

pie_share=[df_train['meter'][df_train.meter==0].count()/df_train['meter'].count(),

           df_train['meter'][df_train.meter==1].count()/df_train['meter'].count(),

           df_train['meter'][df_train.meter==2].count()/df_train['meter'].count(),

           df_train['meter'][df_train.meter==3].count()/df_train['meter'].count()

          ]

figureObject, axesObject = plt.subplots(figsize=(6,6))

pie_colors=('blue','orange','grey','green')

pie_explode=(.05,.05,.15,.05)

axesObject.pie(pie_share,labels=pie_labels,explode=pie_explode,autopct='%.2f%%',colors=pie_colors,startangle=45)

axesObject.axis('equal')

plt.title('TRAIN DATA CATEGORIZATION BY METER TYPE',color='blue',fontsize=12)

plt.show()

#FEATURE SUMMARY BY METER TYPE

print('FEATURE SUMMARY METER TYPE 0')

display(feature_summary(df_train[df_train.meter==0]))
#PLOT METER READING BY DATES FOR METER TYPE 0 

plt.figure(figsize=(50,10))

plt.title("METER READING BY DATES FOR METER TYPE 0",fontsize=40,color='b')

plt.xlabel("Dates",fontsize=40,color='b')

plt.ylabel("Meter Reading",fontsize=40,color='b')

plt.xticks(fontsize=35)

plt.yticks(fontsize=35)

plt.plot(df_train['timestamp'][df_train.meter==0],df_train['meter_reading'][df_train.meter==0],color='blue',linewidth=3)



plt.show()
#FEATURE SUMMARY BY METER TYPE

print('FEATURE SUMMARY METER TYPE 1')

display(feature_summary(df_train[df_train.meter==1]))
#PLOT METER READING BY DATES FOR METER TYPE 1 

plt.figure(figsize=(50,10))

plt.title("METER READING BY DATES FOR METER TYPE 1",fontsize=40,color='b')

plt.xlabel("Dates",fontsize=40,color='b')

plt.ylabel("Meter Reading",fontsize=40,color='b')

plt.xticks(fontsize=35)

plt.yticks(fontsize=35)

plt.plot(df_train['timestamp'][df_train.meter==1],df_train['meter_reading'][df_train.meter==1],color='orange',linewidth=3)



plt.show()
#FEATURE SUMMARY BY METER TYPE

print('FEATURE SUMMARY METER TYPE 2')

display(feature_summary(df_train[df_train.meter==2]))
#PLOT METER READING BY DATES FOR METER TYPE 2 

plt.figure(figsize=(50,10))

plt.title("METER READING BY DATES FOR METER TYPE 2",fontsize=40,color='b')

plt.xlabel("Dates",fontsize=40,color='b')

plt.ylabel("Meter Reading",fontsize=40,color='b')

plt.xticks(fontsize=35)

plt.yticks(fontsize=35)

plt.plot(df_train['timestamp'][df_train.meter==2],df_train['meter_reading'][df_train.meter==2],color='grey',linewidth=3)



plt.show()
#FEATURE SUMMARY BY METER TYPE

print('FEATURE SUMMARY METER TYPE 3')

display(feature_summary(df_train[df_train.meter==3]))
#PLOT METER READING BY DATES FOR METER TYPE 3

plt.figure(figsize=(50,10))

plt.title("METER READING BY DATES FOR METER TYPE 3",fontsize=40,color='b')

plt.xlabel("Dates",fontsize=40,color='b')

plt.ylabel("Meter Reading",fontsize=40,color='b')

plt.xticks(fontsize=35)

plt.yticks(fontsize=35)

plt.plot(df_train['timestamp'][df_train.meter==3],df_train['meter_reading'][df_train.meter==3],color='green',linewidth=3)



plt.show()
#FEATURE SUMMARY FOR BUILDING METADATA DATASET

feature_summary(df_building_metadata)
#CONVERTING timestamp TO DATATIME FIELD IN WEATHER TRAIN DATASET AND EXTRACTING OTHER TIME FEATURES

df_weather_train['timestamp']=pd.to_datetime(df_weather_train['timestamp'])

df_weather_train['month']=df_weather_train.timestamp.dt.month

df_weather_train['year']=df_weather_train.timestamp.dt.year

df_weather_train['day']=df_weather_train.timestamp.dt.day

df_weather_train['hour']=df_weather_train.timestamp.dt.hour

df_weather_train['week_day']=df_weather_train.timestamp.apply(lambda x:x.weekday())

df_weather_train['week']=df_weather_train.timestamp.apply(lambda x:x.isocalendar()[1])

#FEATURE SUMMARY FOR WEATHER TRAIN DATASET

feature_summary(df_weather_train)
%%time

#CONVERTING timestamp TO DATATIME FIELD IN WEATHER TEST DATASET AND EXTRACTING OTHER TIME FEATURES

df_weather_test['timestamp']=pd.to_datetime(df_weather_test['timestamp'])

df_weather_test['month']=df_weather_test.timestamp.dt.month

df_weather_test['year']=df_weather_test.timestamp.dt.year

df_weather_test['day']=df_weather_test.timestamp.dt.day

df_weather_test['hour']=df_weather_test.timestamp.dt.hour

df_weather_test['week_day']=df_weather_test.timestamp.apply(lambda x:x.weekday())

df_weather_test['week']=df_weather_test.timestamp.apply(lambda x:x.isocalendar()[1])

#FEATURE SUMMARY FOR WEATHER TRAIN DATASET

feature_summary(df_weather_test)
#HORIZONTALLY APPENDING WEATHER TRAIN AND TEST

df_weather=pd.concat([df_weather_train,df_weather_test],axis=0,ignore_index=True)

#FEATURE SUMMARY FOR COMBINED WEATER DATASET

feature_summary(df_weather)
#CALCULATING MEANS FOR SITE ID AND WEEK

df_calc_means=df_weather[['site_id','week','air_temperature','cloud_coverage','dew_temperature','precip_depth_1_hr',

            'sea_level_pressure','wind_direction','wind_speed']].groupby(['site_id','week']).mean().reset_index()

cols=['air_temperature','cloud_coverage','dew_temperature','precip_depth_1_hr',

            'sea_level_pressure','wind_direction','wind_speed']



#ROUNDING OFF MEANS TO 2 DECIMAL PLACES

df_calc_means[cols]=df_calc_means[cols].round(2)



#REMOVING NULL VALUES FROM CALCULATED MEANS DATAFRAME

df_calc_means['cloud_coverage'].replace(np.nan,round(df_calc_means['cloud_coverage'].mean(),2),inplace=True)

df_calc_means['precip_depth_1_hr'].replace(np.nan,round(df_calc_means['precip_depth_1_hr'].mean(),2),inplace=True)

df_calc_means['sea_level_pressure'].replace(np.nan,round(df_calc_means['sea_level_pressure'].mean(),2),inplace=True)



print('FEATURE SUMMARY FOR CALCULATED MEANS BY SITE ID AND WEEK')

feature_summary(df_calc_means)
#JOINING TRAIN SET AND BUILDING METADATA

df_train_BM=pd.merge(df_train,df_building_metadata,how='left',on='building_id')

feature_summary(df_train_BM)
%%time

#UNDERSTANDING BUILDING PRIMARY_USER FEATURE

pu_ls=list(df_train_BM['primary_use'].unique())

df_pu=pd.DataFrame(pu_ls,columns=['primary_use'])

df_pu['% Distribution']=df_pu['primary_use'].apply(lambda x:round(df_train_BM['primary_use'][df_train_BM.primary_use==x].count()/

                                                   df_train_BM['primary_use'].count(),4)*100)

df_pu['Number_of_observations']=df_pu['primary_use'].apply(lambda x:df_train_BM['primary_use'][df_train_BM.primary_use==x].count())

df_pu['Avg_consumption']=df_pu['primary_use'].apply(lambda x:round(df_train_BM['meter_reading'][df_train_BM.primary_use==x].mean(),2))

df_pu['Avg_sq_feet']=df_pu['primary_use'].apply(lambda x:round(df_train_BM['square_feet'][df_train_BM.primary_use==x].mean(),2))

df_pu['Consumption_per_sq_feet']=df_pu['primary_use'].apply(lambda x:round(df_train_BM['meter_reading'][df_train_BM.primary_use==x].sum()/

                                                                          df_train_BM['square_feet'][df_train_BM.primary_use==x].sum(),4))

display(df_pu)
%%time



#PIE CHART SHOWING DATA CATEGORIZATION BY METER TYPE

pie_labels=[]

pie_share=[]



for pu in pu_ls:

    pie_labels.append(pu+' : '+str(df_train_BM['primary_use'][df_train_BM.primary_use==pu].count()))

    pie_share.append(df_train_BM['primary_use'][df_train_BM.primary_use==pu].count()/df_train_BM['primary_use'].count())                  

    

figureObject, axesObject = plt.subplots(figsize=(15,15))



pie_explode=(.1,.1,.1,.1,.89,.89,.89,.1,.99,.99,.99,.99,.99,.99,.99,.99)

axesObject.pie(pie_share,labels=pie_labels,explode=pie_explode,autopct='%.2f%%',startangle=45)

axesObject.axis('equal')

plt.title('TRAIN DATA CATEGORIZATION BY PRIMARY USE OF BUILDING',color='blue',fontsize=12)

plt.show()
#UNDERSTANDING RELATIONSHIP BETWEEN primary_use AND meter FEATURES

df=df_train_BM[['building_id','meter','primary_use']].groupby(['meter','primary_use']).count().reset_index()

df.columns=['meter_type','primary_use','observation_count']

display(df)
#VALUES REPLACING NULL VALUES

print('Mean for floor_count is:',round(df_train_BM['floor_count'].mean(),0))

print('Mode for year_built is:',df_train_BM['year_built'].mode()[0])



#REPLACING NULL VALUES

df_train_BM['floor_count'].replace(np.nan,round(df_building_metadata['floor_count'].mean(),0),inplace=True)

df_train_BM['year_built'].replace(np.nan,df_building_metadata['year_built'].mode()[0],inplace=True)



#FEATURE SUMMARY AFTER REPLACING NULL VALUES FOR FEATURES floor_count AND year_built

print('Feature summary after replacing NULL values')

feature_summary(df_train_BM)
# #CREATING DUMMIES FOR pirmary_use FEATURE

# df_train_BMF=pd.concat([df_train_BM,pd.get_dummies(df_train_BM['primary_use'],prefix='pu')],axis=1)

# df_train_BMF.drop('primary_use',axis=1,inplace=True)



# #FEATURE SUMMARY POST DUMMY CREATION

# print('FEATURE SUMMARY AFTER CREATING DUMMIES')

# feature_summary(df_train_BMF)

del df_train

gc.collect()
%%time

# JOINING JOINED(TRAIN,BUILDING METADATA) WITH WEATHER TRAIN SET ON site_id AND timestamp

cols=['site_id','timestamp','air_temperature','cloud_coverage','dew_temperature','precip_depth_1_hr',

            'sea_level_pressure','wind_direction','wind_speed']

df_train_BMW=pd.merge(df_train_BM,df_weather_train[cols],how='left',on=['site_id','timestamp'])

feature_summary(df_train_BMW)
#CLEARING DATAFRAMES

del df_train_BM

gc.collect()
%%time

#EXTRACTING INFORMATION FROM timestamp FEATURE

df_train_BMW['month']=df_train_BMW.timestamp.dt.month

df_train_BMW['day']=df_train_BMW.timestamp.dt.day

df_train_BMW['hour']=df_train_BMW.timestamp.dt.hour

df_train_BMW['week_day']=df_train_BMW.timestamp.apply(lambda x:x.weekday())

df_train_BMW['week']=df_train_BMW.timestamp.apply(lambda x:x.isocalendar()[1])



#GARBAGE COLLECTION

gc.collect()
#FEATURE SUMMARY FOR NEW FEATURES

lfe=['month','day','hour','week_day','week']

print('FEATURE SUMMARY FOR GENERATING CALCULATED FEATURES')

feature_summary(df_train_BMW[lfe])
%%time

#REPLACING NULL VALUES IN WEATHER RELATED FIELDS

for i in range(0,df_calc_means.shape[0]):

    print('replaceing null for site_id: ',df_calc_means.iloc[i,].site_id,' ; week ',df_calc_means.iloc[i,].week,' ; count ',i)

    df_train_BMW[['site_id','week','air_temperature']][(df_train_BMW.site_id==df_calc_means.iloc[i,].site_id) & (df_train_BMW.week==df_calc_means.iloc[i,].week)].replace(np.nan,df_calc_means.iloc[i,].air_temperature,inplace=True)

    df_train_BMW[['site_id','week','cloud_coverage']][(df_train_BMW.site_id==df_calc_means.iloc[i,].site_id) & (df_train_BMW.week==df_calc_means.iloc[i,].week)].replace(np.nan,df_calc_means.iloc[i,].cloud_coverage,inplace=True)

    df_train_BMW[['site_id','week','dew_temperature']][(df_train_BMW.site_id==df_calc_means.iloc[i,].site_id) & (df_train_BMW.week==df_calc_means.iloc[i,].week)].replace(np.nan,df_calc_means.iloc[i,].dew_temperature,inplace=True)

    df_train_BMW[['site_id','week','precip_depth_1_hr']][(df_train_BMW.site_id==df_calc_means.iloc[i,].site_id) & (df_train_BMW.week==df_calc_means.iloc[i,].week)].replace(np.nan,df_calc_means.iloc[i,].precip_depth_1_hr,inplace=True)

    df_train_BMW[['site_id','week','sea_level_pressure']][(df_train_BMW.site_id==df_calc_means.iloc[i,].site_id) & (df_train_BMW.week==df_calc_means.iloc[i,].week)].replace(np.nan,df_calc_means.iloc[i,].sea_level_pressure,inplace=True)

    df_train_BMW[['site_id','week','wind_direction']][(df_train_BMW.site_id==df_calc_means.iloc[i,].site_id) & (df_train_BMW.week==df_calc_means.iloc[i,].week)].replace(np.nan,df_calc_means.iloc[i,].wind_direction,inplace=True)

    df_train_BMW[['site_id','week','wind_speed']][(df_train_BMW.site_id==df_calc_means.iloc[i,].site_id) & (df_train_BMW.week==df_calc_means.iloc[i,].week)].replace(np.nan,df_calc_means.iloc[i,].wind_speed,inplace=True)
#FEATURE SUMMARY POST REPLACING WEATHER NULL VALUES

feature_summary(df_train_BMW)
def ASHRAE_predict_lgb(X,y,i): 

    



    params = {'num_leaves': 31,

              'objective': 'regression',

              'learning_rate': 0.1,

              "boosting": "gbdt",

              "bagging_freq": 5,

              "bagging_fraction": 0.1,

              "feature_fraction": 0.9,

              "metric": 'rmse',

              }



    k=1

    splits=2

    avg_score=0





    kf = KFold(n_splits=splits, shuffle=True, random_state=200)

    print('\nStarting KFold iterations...')

    for train_index,test_index in kf.split(X):



        

        df_X=X[train_index,:]

        df_y=y[train_index]

        val_X=X[test_index,:]

        val_y=y[test_index]



        

        dtrain = lgb.Dataset(df_X, label=df_y)

        dvalid = lgb.Dataset(val_X, label=val_y)

        model=lgb.train(params, dtrain, 10000, valid_sets = [dtrain, dvalid],

                        verbose_eval=2000, early_stopping_rounds=500)



        preds_x=pd.Series(model.predict(val_X))

        preds_x=[x if x>=0 else 0 for x in preds_x]

        acc=rmse(val_y,preds_x)

        print('Iteration:',k,'  rmsle:',acc)

        #SAVING MODEL

        Pkl_Filename = "Pickle_Model_"+str(k)+"_combi_"+str(i)+".pkl"  



        with open(Pkl_Filename, 'wb') as file:

            pickle.dump(model, file)

        print('MODEL SAVED...')

        if k==1:

            score=acc

            preds=pd.Series(preds_x)

            acct=pd.Series(val_y)

        

        else:

            preds=preds.append(pd.Series(preds_x))

            acct=acct.append(pd.Series(val_y))

            if score<acc:

                score=acc

                

        avg_score=avg_score+acc        

        k=k+1

    print('\n Best score:',score,' Avg Score:',avg_score/splits)

#     preds=preds/splits

    return(acct,preds)
%%time

#TRAINING MODELS

for i in range(0,df.shape[0]):

    #SPLITTING TRAINING DATA BY METER TYPE

    print('TRAINING MODEL FOR METER TYPE: ',df.iloc[i,0],' AND PRIMARY USE:',df.iloc[i,1])  

    X=df_train_BMW[(df_train_BMW.meter==df.iloc[i,0]) & (df_train_BMW.primary_use==df.iloc[i,1])].drop(['building_id','timestamp','meter','meter_reading','site_id','primary_use'],axis=1).values

    y=np.log1p(df_train_BMW['meter_reading'][(df_train_BMW.meter==df.iloc[i,0]) & (df_train_BMW.primary_use==df.iloc[i,1])].values)

    

    #FITTING MODEL

    val_y,preds_x=ASHRAE_predict_lgb(X,y,i)

#     print(val_y.shape,preds_x.shape)

    if i==0:

        preds=pd.Series(preds_x)

        acct=pd.Series(val_y)

    else:

        preds=preds.append(preds_x)

        acct=acct.append(val_y)

           

    del X,y

    gc.collect()

    

    

# print(acct.shape,preds.shape)

print('OVER ALL ACCURACY:',rmse(acct,preds))    

    

    
del df_train_BMW

gc.collect()
41697600/20
# #READING SAMPLE SUBMISSION FILE

# submission=pd.read_csv(path1+'sample_submission.csv')
# %%time

# #READING TEST DATA IN CHUNKS

# c_size=2084880

# k=1

# subf=pd.DataFrame()

# for df_test in pd.read_csv(path1+'test.csv',chunksize=c_size):

#     print(df_test.shape)

#     print('Predicting chunk:',k,' of 20')

    

#     df_test['timestamp']=pd.to_datetime(df_test['timestamp'])

    

#     #JOINING WITH BUILDING METADATA

#     df_test_BM=pd.merge(df_test,df_building_metadata,how='left',on='building_id')

    

#     #GARBAGE COLLECTION

#     del df_test

#     gc.collect()

    

#     #REPLACING NULL VALUES

#     df_test_BM['floor_count'].replace(np.nan,round(df_building_metadata['floor_count'].mean(),0),inplace=True)

#     df_test_BM['year_built'].replace(np.nan,df_building_metadata['year_built'].mode()[0],inplace=True)

    

    

#     # JOINING JOINED(TRAIN,BUILDING METADATA) WITH WEATHER TEST SET ON site_id AND timestamp

#     cols=['site_id','timestamp','air_temperature','cloud_coverage','dew_temperature','precip_depth_1_hr',

#             'sea_level_pressure','wind_direction','wind_speed']

#     df_test_BMW=pd.merge(df_test_BM,df_weather_test[cols],how='left',on=['site_id','timestamp'])

    

#     #GARBAGE COLLECTION

#     del df_test_BM

#     gc.collect()

    

#     #EXTRACTING INFORMATION FROM timestamp FEATURE

#     df_test_BMW['month']=df_test_BMW.timestamp.dt.month

#     df_test_BMW['day']=df_test_BMW.timestamp.dt.day

#     df_test_BMW['hour']=df_test_BMW.timestamp.dt.hour

#     df_test_BMW['week_day']=df_test_BMW.timestamp.apply(lambda x:x.weekday())

#     df_test_BMW['week']=df_test_BMW.timestamp.apply(lambda x:x.isocalendar()[1])

    

#     print('Data Preparation Done')

#     for i in range(0,df.shape[0]):

#         sub=pd.DataFrame()

        

#         #SPLITTING TRAINING DATA BY METER TYPE

#         print('PREDICTING FOR METER TYPE: ',df.iloc[i,0],' AND PRIMARY USE:',df.iloc[i,1])  

#         X=df_test_BMW[(df_test_BMW.meter==df.iloc[i,0]) & (df_test_BMW.primary_use==df.iloc[i,1])].drop(['building_id','timestamp','meter','row_id','site_id','primary_use'],axis=1).values

#         sub['row_id']=df_test_BMW['row_id'][(df_test_BMW.meter==df.iloc[i,0]) & (df_test_BMW.primary_use==df.iloc[i,1])].values

        

        

        

#         gc.collect()

        

#         if X.shape[0]!=0:

#             Pkl_Filename1 = "Pickle_Model_"+str(1)+"_combi_"+str(i)+".pkl"  

#             Pkl_Filename2 = "Pickle_Model_"+str(2)+"_combi_"+str(i)+".pkl" 

        

        

#             with open(path+Pkl_Filename1, 'rb') as file:

#                 model1 = pickle.load(file)

        

#             with open(path+Pkl_Filename2, 'rb') as file:

#                 model2 = pickle.load(file)

        

#             sub['meter_reading1']=pd.Series(model1.predict(X))

# #             sub['meter_reading1']=[x if x>=0 else 0 for x in sub['meter_reading1']]

        

#             sub['meter_reading1']=sub['meter_reading1']+pd.Series(model1.predict(X))

#             sub['meter_reading1']=round(sub['meter_reading1'],4)/2

#             sub['meter_reading1']=np.expm1(sub['meter_reading1'])

        

#             subf=pd.concat([subf,sub],axis=0,ignore_index=True)

#             print('Shape of sub predicted chunk:',sub.shape)

#         else:

#             print('No Rows found:',X.shape)

    

#     print('Shape of final predicted chunk(2084880,2):',subf.shape)

#     df_test_BMW

#     gc.collect()

#     k=k+1

# subf['meter_reading1']=[x if x>=0 else 0 for x in subf['meter_reading1']]

# print('Shape of final predicted set (41697600,2):',subf.shape)

# subf.to_csv('sub_initial.csv',index=False)

# subf
# #CREATING SUBMISSION FILE

# submission_f=pd.merge(submission,subf,how='left',on='row_id')

# submission_f.drop('meter_reading',axis=1,inplace=True)

# submission_f.columns=['row_id','meter_reading']

# submission_f.to_csv('submission.csv', index=False)

# submission_f
# %%time

# #JOINING TRAIN SET AND BUILDING METADATA

# df_test_BM=pd.merge(df_test,df_building_metadata,how='left',on='building_id')

# print('AFTER JOINING OF BUILDING METADATA WITH TRAIN SET')

# feature_summary(df_test_BM)
# #VALUES REPLACING NULL VALUES

# print('Mean for floor_count is:',round(df_test_BM['floor_count'].mean(),0))

# print('Mode for year_built is:',df_test_BM['year_built'].mode()[0])



# #REPLACING NULL VALUES

# df_test_BM['floor_count'].replace(np.nan,round(df_test_BM['floor_count'].mean(),0),inplace=True)

# df_test_BM['year_built'].replace(np.nan,df_test_BM['year_built'].mode()[0],inplace=True)



# #FEATURE SUMMARY AFTER REPLACING NULL VALUES FOR FEATURES floor_count AND year_built

# print('Feature summary after replacing NULL values')

# feature_summary(df_test_BM)
# %%time

# # JOINING JOINED(TRAIN,BUILDING METADATA) WITH WEATHER TRAIN SET ON site_id AND timestamp

# cols=['site_id','timestamp','air_temperature','cloud_coverage','dew_temperature','precip_depth_1_hr',

#             'sea_level_pressure','wind_direction','wind_speed']

# df_test_BMW=pd.merge(df_test_BM,df_weather_test[cols],how='left',on=['site_id','timestamp'])

# feature_summary(df_test_BMW)
# #CLEARING DATAFRAMES

# del df_test_BM

# gc.collect()
# 41697600/20
# %%time

# #EXTRACTING INFORMATION FROM timestamp FEATURE

# df_test_BMW['month']=df_test_BMW.timestamp.apply(lambda x:x.month)

# df_test_BMW['day']=df_test_BMW.timestamp.apply(lambda x:x.day)

# df_test_BMW['hour']=df_test_BMW.timestamp.apply(lambda x:x.hour)

# df_test_BMW['week_day']=df_test_BMW.timestamp.apply(lambda x:x.weekday())

# df_test_BMW['week']=df_test_BMW.timestamp.apply(lambda x:x.isocalendar()[1])



# #GARBAGE COLLECTION

# gc.collect()
#JOINING TRAIN SET WITH BUILDING METADATA ON site_id

# df_train_building=df_train.join(df_build_meta,)
# df_test=pd.read_csv(path1+'test.csv',nrows=100000)
# df_test.head()
# df_submission=pd.read_csv(path1+'sample_submission.csv',usecols=['row_id'])
# feature_summary(df_submission)
# df_weather_test=pd.read_csv(path1+'weather_test.csv')

# feature_summary(df_weather_test)