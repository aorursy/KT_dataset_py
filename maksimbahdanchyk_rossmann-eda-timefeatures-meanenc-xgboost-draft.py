import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# This decoratore shows some info about function perormance

# etc time,shpe changes,nan values



def info(function):

    import datetime

    def wrapper(data,*args,**kargs):

        tic    = datetime.datetime.now()

        result = function(data,*args,**kargs)

        toc    = datetime.datetime.now()

        print(function.__name__,' took ', toc-tic)

        print('Shape: ',data.shape,' ----> ', result.shape)

        print('NaN value: ', result.isna().sum()[result.isna().sum() != 0])

        print('\n')

        return result

    return wrapper
# let`s load datasets as usually



train  = pd.read_csv('/kaggle/input/rossmann-store-sales/train.csv')

test   = pd.read_csv('/kaggle/input/rossmann-store-sales/test.csv')

stores = pd.read_csv('/kaggle/input/rossmann-store-sales/store.csv')

sample = pd.read_csv('/kaggle/input/rossmann-store-sales/sample_submission.csv')
# Thank to notebooks we can definve evaluation metric



def ToWeight(y):

    w = np.zeros(y.shape, dtype=float)

    ind = y != 0

    w[ind] = 1./(y[ind]**2)

    return w





def rmspe(yhat, y):

    w = ToWeight(y)

    rmspe = np.sqrt(np.mean( w * (y - yhat)**2 ))

    return rmspe
import seaborn as sns



fig,ax =plt.subplots(1,2,figsize = (20,10))

ins1 = ax[0].inset_axes([0.5,0.5,0.4,0.4])

ins2 = ax[1].inset_axes([0.7,0.7,0.2,0.2])



sns.distplot(train[train.Sales != 0].Sales,ax=ax[0],bins=100) 

sns.distplot(np.log1p(train[train.Sales != 0].Sales),ax=ins1,bins=100,color = 'red')





sns.boxplot(train[train.Sales != 0].Sales,ax=ax[1])

sns.boxplot(np.log1p(train[train.Sales != 0].Sales),ax=ins2)





# We see that sales values shpw positive skeew, it can be fixed by applying np.log1p (embedded plot)

# Also there are some outliers, lets define functions to perform transformation and outliers removal
@info

def log_transf(df):

    # log transformation function to remove skeew

    df.Sales     = np.log1p(df.Sales)

    df.Customers = np.log1p(df.Customers)

    return df



@info

def remove_outliers(df,column='Sales'):

    # interquntile approach to remove outliers

    q1  = df[column].quantile(0.2)

    q3  = df[column].quantile(0.8)

    iqr = q3-q1

    iqr_lower = q1 - 1.5*iqr

    iqr_upper = q3 + 1.5*iqr

    

    df = df.loc[(df[column] > iqr_lower) & (df[column]< iqr_upper),:]

    return df
@info

def timeseries_features(df):

    # move to datetime format

    df.Date = pd.to_datetime(df.Date)

    df = df.sort_values('Date').reset_index(drop = True)

    

    # derive regular for ml task time series features

    df['month']          = df.Date.dt.month

    df['dayofmonth']     = df.Date.dt.day

    df['dayofyear']      = df.Date.dt.dayofyear

    df['year']           = df.Date.dt.year

    df['is_weekday']     = df.DayOfWeek.apply(lambda x: 0 if x in (6,7) else 1)

    df['is_month_start'] = df.Date.dt.is_month_start.astype(int)

    df['is_month_end']   = df.Date.dt.is_month_end.astype(int)



    # also lets take into account holidays

    from pandas.tseries.holiday import USFederalHolidayCalendar as calendar



    holidays = calendar().holidays(start = df.Date.min(), end = df.Date.max())

    df['is_holiday'] = df.Date.isin(holidays).astype(int)

    



    return df



@info

def clean_main(df):

    # drop days with 0 sales

    df = df.loc[df.Sales != 0,:].reset_index(drop = True)

    df = df.drop(['Open'],axis = 1)

    

    # beacus unique values contain mixed dtype array(['a', '0', 'b', 'c', 0], dtype=object)

    # also could be fixed during pandas importing

    df.StateHoliday = df.StateHoliday.astype(str)

    

    return df



@info

def clean_store(df):

    # lets drop columns with high content of nan values

    df.CompetitionDistance.fillna(df.CompetitionDistance.mean(),inplace = True)

    df.drop(['CompetitionOpenSinceMonth','CompetitionOpenSinceYear','Promo2SinceWeek','Promo2SinceYear'],axis = 1,inplace = True)

    

    import calendar

    

    # We have a list of promo monthes and we can derive usefull feature

    # presence or absence of promo

    

    # first create encoded dictionary Month:Number eg Feb:2

    month_dict = {v: k for k,v in enumerate(calendar.month_abbr)}

    del month_dict['']

    del month_dict['Sep']

    month_dict['NaN']   = 0 # assign absence of promo 0

    month_dict['Sept']  = 9 # There is no Sep



    # Secondly, we treat PromoInterval columns, making each row list instead of string  now we have smth like ['Feb','Mar','Sept']

    # and lets apply dictionary

    df.PromoInterval = df.PromoInterval.fillna('NaN')

    df.PromoInterval = df.PromoInterval.str.split(',')

    # Lastly we are applyin transformation

    df.PromoInterval = df.PromoInterval.apply(lambda x: [month_dict[value] for value in x if month_dict.get(value)])

    

    # Lets create new feature that us equal to number of promo monthes

    df['promo_len']  = df.PromoInterval.apply(lambda x: len(x))

    return df





# Pipeline for train file

train_prep = (train

             .copy()

             .pipe(log_transf)

             .pipe(remove_outliers)

             .pipe(timeseries_features)

             .pipe(clean_main)

             )



# Pipeline for store file

store_prep = (stores

             .copy()

             .pipe(clean_store)

             )

# Now we merge two

data_prep               = pd.merge(train_prep,store_prep,how='left',on='Store')



# Using our transformation in PromoInterval interval, we create binary new feature is_promo or not

data_prep['is_promo']   = data_prep.apply(lambda x: 1 if x['month'] in x['PromoInterval'] else 0,axis = 1)

data_prep               = data_prep.drop('PromoInterval',axis=1).reset_index(drop=True)
# Here I would like to know what is rmspe score with th emost dumb approach



# we devide data on train and test

test_bs  = data_prep[data_prep.year == 2015]

train_bs = data_prep[data_prep.year  < 2015]



# I am going to use mean Sales grouped by store-month-day among previous years as predicted values for 2015

predict_bs = (train_bs

              .groupby(['Store','month','dayofmonth']).Sales.mean().reset_index().rename({'Sales':'predictions'},axis = 1)

              .merge(test_bs,how='right',on = ['Store','month','dayofmonth'])

              .fillna(train_bs.Sales.mean())

              .sort_values('Date')

              )



# Display baseline

print('Baseline to overcome = {:.2f}'.format(rmspe(np.expm1(predict_bs.Sales),np.expm1(predict_bs.predictions))))



# Let`s see how prediction looks like

fig,ax = plt.subplots(1,3,figsize = (30,10))



rnd_store = np.random.randint(min(predict_bs.Store),max(predict_bs.Store),3)



for idx,store in enumerate(rnd_store):

    

    ax[idx].plot(predict_bs[predict_bs.Store == store].Date,np.expm1(predict_bs[predict_bs.Store == store].Sales), color = 'blue'    ,label = 'Observed')

    ax[idx].plot(predict_bs[predict_bs.Store == store].Date,np.expm1(predict_bs[predict_bs.Store == store].predictions),color = 'red',label = 'Predicted')



    ax[idx].legend()

    ax[idx].set_title('Store '+str(store))

    

    

# It doesn`t look so bad
# There are two few reasons to use mean (aka target) encoding

# We have 1115 stores, definetly there is correlation between store and sales

# We could perform leave stores as it is ----> not good for known reasons

# We could perform OneHotEncoding        ----> not goodm becaouse we will have 1115 new columns, mainly sparse

# We can do mean encoding, eg encode stores as mean/std/other of target

# I am going to use Customers to encode store, because we don`t have customers in test set

# Obviusly customers can be good feature



def mean_encoding(df,column,target,func = np.mean):

    

    # perform target encoding on column with some function

    

    enc_col_name = target+'_enc_'+func.__name__

    df_temp = (df

               .groupby(column)[target]

               .apply(func)

               .reset_index()

               .rename({target:enc_col_name},axis=1)

              )

    

    df = df.merge(df_temp,how='left',on = column)

        

    

    return df,df_temp



data_prep,dict_for_test = mean_encoding(data_prep,'Store','Customers',func = np.mean) 
# also it is good to statistic



from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

from statsmodels.tsa.stattools import adfuller





# first lets check our data for stationarity



counter = 0

for store in data_prep.Store.unique():

    df_store = data_prep.copy().loc[data_prep.Store == store,['Date','Sales']].set_index('Date')

    # since we removed some dates, lets resample data on a daily basis and fillna with 0

    df_store = df_store.resample('D').fillna('bfill')

    adf = adfuller(df_store,regression='c', autolag='AIC')

    

    if adf[1] > 0.05:

        print('Adfuller for store {} : p-value = {:.5f} > 5% -----> NON STATIONARY'.format(store,adf[1]*100))

        counter+=1

        # also we can use it as a feature

        # Doesnt make sense becaause only ~3 of store are not statonary

        

print('\n {:.2f} % of stores are non stationary '.format(counter/len(data_prep.Store.unique())*100))



# There is a chance to use traditional time series technique(ARIMA,SARIMAX, smothing) but i would ike to continue with ml
# lets check few random stores 



rnd_store = np.random.randint(min(data_prep.Store),max(data_prep.Store),3)



fig,ax = plt.subplots(3,2,figsize = (15,10))



for idx,store in enumerate(rnd_store):

    df_store = data_prep.copy().loc[data_prep.Store == store,['Date','Sales']].set_index('Date')

    df_store = df_store.resample('D').fillna('bfill')

    plot_acf(df_store,lags = 60,ax = ax[idx,0],label = store) 

    plot_pacf(df_store,lags = 60,ax = ax[idx,1], label = store)

    ax[idx,0].set_title('Autocorelation for store {}'.format(store))

    ax[idx,1].set_title('Partial Autocorelation for store {}'.format(store))

    plt.tight_layout()

    

    

# By running this part few times we can notice that almost for all stores there is hogh corelation with following lags:

# 1 14,28,42, 49

# Therefore lets use this values to create new features

# But we need to preduct 48 days in future, threre fore we cannot use something lower 48
def lag_creator(df,lags = [1,14,28,42,49],col = 'Sales',mean = False,std = False,rolling = 90):

    #  we can create lags if we want

    for lag in lags:

        col_tag = 'lag-'+str(lag)

        df[col_tag] = df.groupby(['Store'])[col].shift(lag).values

        if mean:

            col_tag_mean = 'lag_mean-'+str(lag)

            df[col_tag_mean] = df.groupby(['Store'])[col].shift(lag).rolling(window = rolling).mean().values

        elif std:

            col_tag_std= 'lag_std-'+str(lag)

            df[col_tag_mean] = df.groupby(['Store'])[col].shift(lag).rolling(window = rolling).std().values

        elif mean==True and std == True:

            col_tag_mean = 'lag_mean-'+str(lag)

            df[col_tag_mean] = df.groupby(['Store'])[col].shift(lag).rolling(window = rolling).mean().values

            col_tag_std= 'lag_std-'+str(lag)

            df[col_tag_mean] = df.groupby(['Store'])[col].shift(lag).rolling(window = rolling).std().values

    else: 

        print('No statistics lags')

            

    return df.dropna().reset_index(drop = True)



#data_prep = data_prep.pipe(lag_creator)

# We won`t use lags
# finally lets check on nan and dublicated values

 

print('NaN summary\n\n',data_prep.isna().sum()/len(data_prep)*100,'\n')

print('Number of absoulute    dublicates:',data_prep.duplicated().sum())

print('Number of Store - Date dublicates:',data_prep.duplicated(subset = ['Date','Store']).sum())
from pandas.plotting import scatter_matrix

import seaborn



corr = data_prep.corr()

plt.figure(figsize=(15,15))

seaborn.heatmap(corr)
stores = np.random.randint(train.Store.min(),train.Store.max(),2)

plt.figure(figsize=(15,10))

for store in stores:

    plt.plot(data_prep.loc[(data_prep.Store == store) & (data_prep.year == 2013),'Date'],data_prep.loc[(data_prep.Store == store) & (data_prep.year == 2013),'Sales'],label = store)

    plt.legend()
ohe_col = data_prep.select_dtypes('object').columns.tolist()+['Store','DayOfWeek','month']

num_col = data_prep.select_dtypes('float').columns.tolist()
X = data_prep.drop(['Date','Sales','Customers','Store'],axis = 1)

y = data_prep.Sales





X_train,X_val = X.loc[X.year < 2015,:],X.loc[X.year == 2015,:]

y_train,y_val = y[:X_train.index[-1]+1], y[X_train.index[-1]+1:]
from sklearn.preprocessing import StandardScaler,MinMaxScaler

from sklearn.preprocessing import OneHotEncoder,LabelEncoder,OrdinalEncoder

from sklearn.compose import make_column_transformer

from sklearn.pipeline import make_pipeline



transformer = make_column_transformer(

    (StandardScaler(),['CompetitionDistance', 'Customers_enc_mean']),

    (OneHotEncoder(),['StateHoliday', 'StoreType', 'Assortment', 'DayOfWeek', 'month']),

    remainder = 'passthrough'

)





import xgboost as xgb



regressor = xgb.XGBRegressor(n_estimators = 200,

                             max_depth    =  10

                            )





pipeline = make_pipeline(transformer,

                         regressor)





pipeline.fit(X_train,y_train)





print('TRAIN RMSPE = ',rmspe(np.expm1(pipeline.predict(X_train)),np.expm1(y_train)))

print('VAL   RMSPE = ',rmspe(np.expm1(pipeline.predict(X_val)),np.expm1(y_val)))
# We need to apply same transformation on test set as we did we train set



test_prep = (test

             .copy()

             .pipe(timeseries_features)

             .drop(['Open','Date'],axis=1)

             )



test_prep  = pd.merge(test_prep,store_prep,how='left',on='Store')

test_prep['is_promo']   = test_prep.apply(lambda x: 1 if x['month'] in x['PromoInterval'] else 0,axis = 1)

test_prep  = pd.merge(test_prep,dict_for_test,how='left',on='Store')

test_prep = test_prep.drop(['PromoInterval','Store'],axis=1).reset_index(drop=True)

test_id = test_prep.Id

test_prep.drop('Id',axis=1,inplace = True)

predict = np.expm1(pipeline.predict(test_prep)) # Remember to make inverse transformation
sub = pd.DataFrame({'Id':test_id,'Sales':predict}).sort_values('Id').reset_index(drop=True)

sub.to_csv('submission.csv',index=False)