import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import pandas_profiling 

import math

import numpy as np

import gc

from pandas.api.types import CategoricalDtype

from scipy.special import boxcox1p

%matplotlib inline

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

sns.set(rc={'figure.figsize':(11.7,8.27)})

import warnings

warnings.filterwarnings(action='once')
import zipfile

zip_paths = ['/kaggle/input/walmart-recruiting-store-sales-forecasting/features.csv.zip','/kaggle/input/walmart-recruiting-store-sales-forecasting/test.csv.zip','/kaggle/input/walmart-recruiting-store-sales-forecasting/sampleSubmission.csv.zip','/kaggle/input/walmart-recruiting-store-sales-forecasting/train.csv.zip']

directory_to_extract_to = '/kaggle/working'

for path_to_zip_file in zip_paths:

    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:

        zip_ref.extractall(directory_to_extract_to)
for dirname, _, filenames in os.walk('/kaggle/working'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
rtrain = pd.DataFrame(pd.read_csv('/kaggle/working/train.csv',parse_dates=[2],index_col=2,squeeze=True))

rfeatures = pd.DataFrame(pd.read_csv('/kaggle/working/features.csv',parse_dates=[1],index_col=1,squeeze=True))

rtest = pd.DataFrame(pd.read_csv('/kaggle/working/test.csv',parse_dates=[2],index_col=2,squeeze=True))

rstore = pd.DataFrame(pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/stores.csv'))
rtrain
# plot all the sales at once

sns.lineplot(data=rtrain,x=rtrain.index, y="Weekly_Sales")
# following takes awefull amount of time and very unclear to see proper patterns. so we plot just a slice of it in below cell.

sns.lineplot(data=rtrain,x=rtrain.index, y="Weekly_Sales",hue="Store")
storeid = [val*5 for val in range(1,10)]

tmp_rtrain = pd.DataFrame(rtrain.loc[rtrain['Store'].isin(storeid)])

tmp_rtrain['Store'].unique()
# we plot weekly sales of above stores to look in depth

sns.lineplot(data=tmp_rtrain,x=tmp_rtrain.index, y="Weekly_Sales",hue="Store",ci=None)
tmp_rtrain = pd.DataFrame(rtrain.loc[rtrain['Store']==1])

tmp_rtrain['Store'].unique()                          
sns.lineplot(data=tmp_rtrain,x=tmp_rtrain.index, y="Weekly_Sales",hue="Dept",ci=None)
tmp_rtrain = pd.DataFrame(rtrain.loc[rtrain['Store']==2])

print (tmp_rtrain['Store'].unique())

sns.lineplot(data=tmp_rtrain,x=tmp_rtrain.index, y="Weekly_Sales",hue="Dept",ci=None)
storeid = rtrain['Store'].unique()

print (len(storeid))

print (len(rtrain['Dept'].unique()))

print (45*81)
# utilities class 

class Utils:

    @classmethod

    def check_holiday_same(cls, rtrain_orig, rfeatures_orig, storeid):

        '''

        This checks the IsHoliday feature of training data is same as IsHoliday feature of features data

        '''

        rtrain = rtrain_orig.copy()

        rfeatures = rfeatures_orig.copy()

        tmp_rf = rfeatures.loc[rfeatures['Store']==storeid]

        tmp_df = rtrain.loc[(rtrain['Store']==storeid) & (rtrain['Dept']==1)]

        tmp_df = tmp_df.rename(columns={'IsHoliday':'storeHoliday'})

        tmp_df2 = pd.merge(tmp_df,tmp_rf, left_index=True,right_index=True)

        val = tmp_df2['storeHoliday'].equals(tmp_df2['IsHoliday'])

        return val

    

    @classmethod

    def missing_vals(cls, indf, id_str=None):

        if (id_str is None):

            id_str = 'Id'

        countdf = indf.count()

        missdict = {}

        for key,val in countdf.items():

            missdict[key] = countdf[id_str] - val

        missdf = pd.DataFrame(missdict.items(),columns=['name','miss_val'])

        miss_pct = pd.DataFrame((missdf['miss_val']/countdf[id_str])*100)

        miss_pct = miss_pct.rename(columns={'miss_val':'miss_pct'})

        missdf = pd.concat([missdf,miss_pct],axis=1,join='inner')

        missdf = missdf.sort_values(by='miss_pct',ascending=False)

        return missdf

    

    @classmethod

    def apply_boxcox1p(cls, indf, collist,lmd):

        temp = indf.copy()

        df = pd.DataFrame(boxcox1p(temp[collist],lmd))

        temp = temp.drop(columns=collist,axis=1)

        outdf = pd.concat([temp,df],join="inner",axis=1)

        return outdf

    

    @classmethod

    def get_dummies(cls, indf, collist):

        tmp = indf.copy()

        for col in collist:

            dummy = pd.get_dummies(indf[col],prefix=col)

            tmp = pd.concat([tmp, dummy],axis=1,join='inner')

            tmp.drop(columns=[col],axis=1,inplace=True)

        return tmp
# check whether IsHoliday in features.csv is duplicate of IsHoliday in train data.

tmp_list = [Utils.check_holiday_same(rtrain, rfeatures, sid) for sid in range(1,46)]

'False' in tmp_list
rfeatures.drop(columns='IsHoliday', inplace=True, axis=1)
# we should combine features data, store data, into train and test data.

# merge with index and column is not working. so we are pulling index and making a column and 

# then merging with two columns (data, store)



temp_train = rtrain.copy()

temp_test = rtest.copy()

temp_feat = rfeatures.copy()

temp_train['Date_col'] = temp_train.index

temp_feat['Date_col'] = temp_feat.index

temp_test['Date_col'] = temp_test.index



temp_train = temp_train.merge(temp_feat,how='left')

temp_test = temp_test.merge(temp_feat,how='left')



temp_train = pd.merge(temp_train,rstore, on=['Store'])

temp_test = pd.merge(temp_test,rstore, on=['Store'])



temp_train.rename(columns={'Date_col':'Date'},inplace=True)

temp_train.index = temp_train['Date']

temp_train.drop('Date',axis=1,inplace=True)



temp_test.rename(columns={'Date_col':'Date'},inplace=True)

temp_test.index = temp_test['Date']

temp_test.drop('Date',axis=1,inplace=True)



rtrain = temp_train

rtest = temp_test

del temp_train

del temp_test

# del rfeatures

# del rstore



gc.collect()
rtrain.loc[rtrain.IsHoliday == True,'IsHoliday'] =1 

rtrain.loc[rtrain.IsHoliday == False,'IsHoliday'] =0



rtest.loc[rtest.IsHoliday == True,'IsHoliday'] =1 

rtest.loc[rtest.IsHoliday == False,'IsHoliday'] =0
# lets check missing values of train, test



miss_train = Utils.missing_vals(rtrain,"Store")

miss_test = Utils.missing_vals(rtest,"Store")
miss_train
miss_test
dp_list = ['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5']

rtrain.drop(columns=dp_list, axis=1,inplace=True)

rtest.drop(columns=dp_list, axis=1,inplace=True)
rtest
# rtest[rtest['CPI'].isnull()].head(40) tells us that in every (store, dept) some equal percent of 

# values are missing in 'CPI' and in 'Unemployment'.

# so we are filling those missing values as avg of rest of the rows in each of (store, dept)



itr_gp = pd.DataFrame(rtest.groupby(['Store','Dept']))

tmp_gp = rtest.groupby(['Store','Dept'])

# ttmp = tmp_gp.get_group((1,1)).mean()



pd.set_option('mode.chained_assignment', None)



for row in itr_gp[0]:

    storeid = row[0]

    deptid = row[1]

    ttmp = tmp_gp.get_group((storeid,deptid)).mean()

    cpival = ttmp['CPI']

    unemp = ttmp['Unemployment']

    rtest.loc[(rtest.Store == storeid) & (rtest.Dept==deptid) & (rtest.CPI.isnull()), 'CPI'] = cpival

    rtest.loc[(rtest.Store == storeid) & (rtest.Dept==deptid) & (rtest.Unemployment.isnull()), 'Unemployment'] = unemp



pd.set_option('mode.chained_assignment', 'raise')
miss_test = Utils.missing_vals(rtest,"Store")

miss_test
rtest[(rtest['Store']==4) & (rtest['Dept']==39)]
tmp_gp2 = rtest.groupby(['Store'])

for row in itr_gp[0]:

    storeid = row[0]

    deptid = row[1]

    ttmp = tmp_gp2.get_group(storeid).mean()

    cpival = ttmp['CPI']

    unemp = ttmp['Unemployment']

    rtest.loc[(rtest.Store == storeid)  & (rtest.CPI.isnull()), 'CPI'] = cpival

    rtest.loc[(rtest.Store == storeid) & (rtest.Unemployment.isnull()), 'Unemployment'] = unemp

# to remove heteroscedasticity, we should apply log transformation. 

collist = ['Weekly_Sales','Temperature','CPI','Unemployment','Size']

rtrain = Utils.apply_boxcox1p(rtrain,collist,0)

collist = ['Temperature','CPI','Unemployment','Size']

rtest = Utils.apply_boxcox1p(rtest,collist,0)
rtrain = Utils.get_dummies(rtrain, ['Type'])

rtest = Utils.get_dummies(rtest, ['Type'])
rtrain.replace([np.inf,-np.inf],np.nan,inplace=True)

rtest.replace([np.inf,-np.inf],np.nan,inplace=True)
# checked where, when temp is null then changing accordingly.

# droping all the NaNs for weekly_sales

# tmp = rtrain.loc[rtrain['Temperature'].isnull()]

# tmp

tmp2 = rtrain.loc['2011-02-11',['Store','Dept','Temperature']]

val = tmp2.loc[tmp2['Store']==7].Temperature.unique()

val = val[0]

rtrain.loc[rtrain['Temperature'].isnull(),'Temperature'] = val

rtrain.dropna(axis=0,inplace=True)
miss_train = Utils.missing_vals(rtrain,"Store")

miss_train
# we got null temp on 13-01-04 (d1), 13-01-11(d2),13-01-18 (d3) at store 7. 

# so we are replacing them with nearby temp of store 7.



before_temp = rtest.loc[(rtest.index=='2012-12-28') & (rtest.Store == 7),'Temperature'][0]

after_temp = rtest.loc[(rtest.index=='2013-01-25') & (rtest.Store == 7),'Temperature'][0]



rtest.loc[(rtest.index=='2013-01-04') & (rtest.Store == 7),'Temperature'] = before_temp

rtest.loc[(rtest.index=='2013-01-11') & (rtest.Store == 7),'Temperature'] = (before_temp+after_temp)/2

rtest.loc[(rtest.index=='2013-01-18') & (rtest.Store == 7),'Temperature'] = after_temp
miss_test = Utils.missing_vals(rtest,"Store")

miss_test
# IsHoliday is of type 'object' lets convert that into int

rtrain['IsHoliday'] = rtrain.IsHoliday.astype('int')

rtest['IsHoliday'] = rtest.IsHoliday.astype('int')
# some Weekly_sales are -ve. so lets make them min i.e. 0

# sales can be -ve and it can affect future or past sales. So keeping it.

# rtrain.loc[rtrain['Weekly_Sales']<0,'Weekly_Sales'] =0
rtrain.describe()
rtest.describe()
sample_sid_did = [[1,1],[2,2],[3,3],[4,4],[5,5]]

sample_sid = [row[0] for row in sample_sid_did]

sample_did = [row[1] for row in sample_sid_did]



sample_train = rtrain.loc[(rtrain.Store == 1) & (rtrain.Dept ==1)]

for row in sample_sid_did:

    if ((row[0]==1) & (row[1]==1)):

        continue

    tmp = rtrain.loc[(rtrain.Store == row[0]) & (rtrain.Dept ==row[1])]

    sample_train = pd.concat([sample_train,tmp])
# Weekly_sales vs predictors 



sns.lineplot(data=sample_train, x=sample_train.index, y='Temperature',hue='Store',palette="Set1")
fig, axes = plt.subplots(2,2,figsize=(15,8))

ax1, ax2, ax3, ax4 = axes.flatten()



sns.lineplot(data=sample_train, x=sample_train.index, y='Fuel_Price',hue='Store',palette="Set1",ax=ax1)

sns.lineplot(data=sample_train, x=sample_train.index, y='CPI',hue='Store',palette="Set1",ax=ax2)

sns.lineplot(data=sample_train, x=sample_train.index, y='Unemployment',hue='Store',palette="Set1",ax=ax3)

sns.lineplot(data=sample_train, x=sample_train.index, y='Size',hue='Store',palette="Set1",ax=ax4)



# ax2.set_ylabel("CPI") 

plt.show()
# correlation between predictors

temp_df = sample_train.loc[(sample_train.Store==1)& (sample_train.Dept==1)]

dcorr = temp_df.corr()

sns.heatmap(dcorr,cmap="coolwarm",annot=True)
# autocorrelation plot of 'Weekly_sales'

from pandas.plotting import autocorrelation_plot



for row in sample_sid_did:

    weekly_sales = rtrain.loc[(rtrain.Store==row[0]) & (rtrain.Dept==row[1]),'Weekly_Sales']

#     weekly_sales = weekly_sales.diff()

#     weekly_sales.dropna(inplace=True)

    ax = autocorrelation_plot(weekly_sales)

    ax.set_xticks(np.arange(0, 140, 4))

plt.grid()

plt.show()
# lets check the trend and seasonality of the sample data using seasonal_decompose



# ADF test to check the stationarity of sample data



import statsmodels.api as sm

from statsmodels.tsa.stattools import adfuller



for row in sample_sid_did:

    sid=row[0]

    did=row[1]

    weekly_sales = sample_train.loc[(sample_train.Store==sid)& (sample_train.Dept==did)].Weekly_Sales

    result = adfuller(weekly_sales,autolag='AIC')

    print ("store="+str(sid)+", deptid="+str(did))

    print ('ADF statistic: %f' % result[0])

    print ('p-value: %f'%result[1])

    print ('critical values')

    for key, val in result[4].items():

        print ('\t%s: %.3f'%(key,val))



    if (result[0]<result[4]['5%']):

        print ("----> TS is stationary")

    else:

        print ("----> TS is not stationary")
# acf and pcaf on original data.

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf



warnings.filterwarnings(action='ignore')

numlags=130

for row in sample_sid_did:

    sid=row[0]

    did=row[1]

    weekly_sales = sample_train.loc[(sample_train.Store==sid)& (sample_train.Dept==did)].Weekly_Sales

    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(20,4))

#     ax1.plot(weekly_sales);

    plot_acf(weekly_sales, ax=ax1,lags=numlags)

    ax1.set_xticks(np.arange(0, 140, 4))

    ax2.set_xticks(np.arange(0, 140, 4))

    plot_pacf(weekly_sales, ax=ax2,lags=numlags)

    plt.show()



warnings.filterwarnings(action='once')
# lag1 acf, pacf plots of timeseries.

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf



numlags=70

for row in sample_sid_did:

    sid=row[0]

    did=row[1]

    weekly_sales = sample_train.loc[(sample_train.Store==sid)& (sample_train.Dept==did)].Weekly_Sales

    weekly_sales = weekly_sales.diff()

    fig, (ax1, ax2,ax3) = plt.subplots(1, 3,figsize=(20,4))

    ax3.plot(weekly_sales);

    plot_acf(weekly_sales, ax=ax1,lags=numlags)

    ax1.set_xticks(np.arange(0, 140, 4))

    ax2.set_xticks(np.arange(0, 140, 4))

    plot_pacf(weekly_sales, ax=ax2,lags=numlags)

    plt.show()


import statsmodels.api as sm

from statsmodels.tsa.stattools import adfuller

i=0

itr_gp = pd.DataFrame(rtest.groupby(['Store','Dept']))

sample_stores = itr_gp[0]

tmp_list = []

error_cnt=0

sample_count=200

for row in sample_stores:

    if (i==sample_count):

        break

    i=i+1

    sid=row[0]

    did=row[1]

    weekly_sales = rtrain.loc[(rtrain.Store==sid)& (rtrain.Dept==did)].Weekly_Sales

    weekly_sales = weekly_sales.diff()

    weekly_sales = weekly_sales.dropna()

    try:

        result = adfuller(weekly_sales,autolag='AIC')

    except:

        error_cnt=error_cnt+1

        continue

#     print ("store="+str(sid)+", deptid="+str(did))

#     print ('ADF statistic: %f' % result[0])

#     print ('p-value: %f'%result[1])

#     print ('critical values')

#     for key, val in result[4].items():

#         print ('\t%s: %.3f'%(key,val))

#     if (result[0]<result[4]['5%']):

#         print ("----> TS is stationary")

#     else:

#         print ("----> TS is not stationary")

    tmp_list.append(result[0]<result[4]['5%'])

    

val = (sum(tmp_list)/(sample_count-error_cnt))

print ("%f percent of samples in %d are stationary"%(val*100,sample_count))
# lets see the graphs of each timeseires and their lag1.



fig, axes = plt.subplots(5,2,figsize=(20,18))



i=0

for row in sample_sid_did:

    sid=row[0]

    did=row[1]

    weekly_sales = rtrain.loc[(rtrain.Store==sid)& (rtrain.Dept==did)].Weekly_Sales

    sns.lineplot(data=weekly_sales, ax=axes[i,0])

    weekly_sales_lag1 = weekly_sales.diff()

    sns.lineplot(data=weekly_sales_lag1, ax=axes[i,1])

    i=i+1

# ax2.set_ylabel("CPI") 

plt.show()
itr_gp = pd.DataFrame(rtest.groupby(['Store','Dept']))

sample_stores = itr_gp[0]

i=0

warnings.filterwarnings(action='ignore')

for row in sample_stores:

    weekly_sales = rtrain.loc[(rtrain.Store==row[0]) & (rtrain.Dept==row[1]),'Weekly_Sales']

    weekly_sales = weekly_sales.diff()

    weekly_sales.dropna(inplace=True)

    ax = autocorrelation_plot(weekly_sales)

    ax.set_xticks(np.arange(0, 140, 4))

    if (i==10):

        break

    i=i+1

    plt.grid()

    plt.show()



warnings.filterwarnings(action='once')
#lets see the data points for depts of each store.

itr_gp = pd.DataFrame(rtest.groupby(['Store','Dept']))

sample_stores = itr_gp[0]

data_points = []

for row in sample_stores:

    cnt = rtrain.loc[(rtrain.Store==row[0]) & (rtrain.Dept==row[1])].Store.count()

    data_points.append(cnt)

data_points = pd.DataFrame(data_points,columns=['points'])
data_points.loc[data_points.points<52]
# we are using auto_arima to get best possible values and we cross check them with ours.

# "pip install pmdarima" to install pmdarima using console in kaggle notebook

# import pmdarima as pm

# dont know why it says no module named 'pmdarima' even after installing by pip



# so we use manual sarima model.
# following is the official error prediction formula.

# weighted mean absolute error (weight =5 on holidays)



def WMAE(dataset, real, predicted):

    weights = dataset.IsHoliday.apply(lambda x: 5 if x else 1)

    return np.round(np.sum(weights*abs(real-predicted))/(np.sum(weights)), 2)
# mtrain is 70% data and mvalid is 30%.

mtrain = rtrain['2010-02-05':'2012-02-10']

mvalid = rtrain['2012-02-17':]
from statsmodels.tsa.statespace.sarimax import SARIMAX

import time



non_seasonal_order=(0,1,1)

myseasonal_order=(1,0,0,52)

sid=1

did=1

exo_features = ['Store', 'Dept', 'IsHoliday', 'Fuel_Price',

       'Temperature', 'CPI', 'Unemployment', 'Size', 'Type_A', 'Type_B',

       'Type_C']

mtrain_sales = mtrain.loc[(mtrain.Store==sid) & (mtrain.Dept==did),'Weekly_Sales']

mtrain_exo = mtrain.loc[(mtrain.Store==sid) & (mtrain.Dept==did),exo_features]



start_time = time.time()

model = SARIMAX(mtrain_sales,order=non_seasonal_order, seasonal_order=myseasonal_order,

                enforce_stationarity=False, enforce_invertibility=False,exogenous=mtrain_exo)

model_fit = model.fit()

end_time = time.time()

print ('Model fit done in: '+str(end_time-start_time)+" sec")
print (model_fit.summary())
import statsmodels.api as sm

train_resid = model_fit.resid

fig,ax = plt.subplots(2,1,figsize=(15,8))

fig = sm.graphics.tsa.plot_acf(train_resid, lags=50, ax=ax[0])

fig = sm.graphics.tsa.plot_pacf(train_resid, lags=50, ax=ax[1])

plt.show()
mvalid_exo = mvalid.loc[(mvalid.Store==sid) & (mvalid.Dept==did),exo_features]

mvalid_actual_result = mvalid.loc[(mvalid.Store==sid) & (mvalid.Dept==did),'Weekly_Sales']

forecast_vals = model_fit.predict(start='2012-02-17',end='2012-10-26',exog=mvalid_exo)

predictions = pd.DataFrame(forecast_vals, index=mvalid_exo.index,columns=['pred'])

valid_residuals = mvalid_actual_result - predictions.pred
sns.lineplot(y=predictions.pred, x=predictions.index,legend='brief',label='pred')

sns.lineplot(y=mvalid_actual_result,x=predictions.index, legend='brief',label='actual')
sns.lineplot(y=valid_residuals,x=predictions.index)
val = WMAE(mvalid_exo, real=mvalid_actual_result,predicted=predictions.pred)

val
# lets create a method to perform modelling for each store, dept combination.

from scipy.special import inv_boxcox



def sarimax_driver(tot_trdata, train_data_pct,ftest=None):

    itr_gp = pd.DataFrame(tot_trdata.groupby(['Store','Dept']))

    all_stores = itr_gp[0]

    exo_features = ['Store', 'Dept', 'IsHoliday', 'Fuel_Price',

           'Temperature', 'CPI', 'Unemployment', 'Size', 'Type_A', 'Type_B',

           'Type_C']

    fresult = pd.DataFrame(columns=['Id','Weekly_Sales'])

    val_result = pd.DataFrame(columns=['sales','Store','Dept'])

    val_result.index.names = ['Date']

    i=0

    for row in all_stores:

        sid = row[0]

        did = row[1]

#         if (i==10):

#             break

#         i+=1

#         if (not ((sid==1) & (did==51))):

#             continue

        cnt = tot_trdata.loc[(tot_trdata.Store==sid) & (tot_trdata.Dept==did)].Store.count()

        # if there are very less observations(10 which is less than 10% of most of (sid,did)pairs (143)), it is not a good idea to pred

        if (cnt<=10):

            continue

        trcnt = int(cnt*train_data_pct)

        valcnt = cnt - trcnt

        store_data = pd.DataFrame(tot_trdata.loc[(tot_trdata.Store==sid) & (tot_trdata.Dept==did)])

        trdata = store_data[0:trcnt]

        valdata = store_data[trcnt:].copy()

        trsales = trdata.Weekly_Sales

        trexo = trdata[exo_features]

        if (ftest is None):

#             print (str(sid)+" "+str(did)+"\t",end="")

            valexo = valdata[exo_features]

            # since some of observations are missing in the validation data. We can not use datetime in start, end. we have to use start, end row number.

#             start_row= valdata.index[0].strftime('%Y-%m-%d')

#             end_row = valdata.index[len(valdata)-1].strftime('%Y-%m-%d')

            start_row = trcnt

            end_row = cnt-1

            predictions = sarimax_specific(trsales, trexo, valexo, start_row, end_row)

            if (predictions is None):

                print ("Exception in train/val (sid, did)"+str(sid)+" "+str(did))

                continue

            preds = pd.DataFrame(data=predictions, columns=['sales'])

            preds.index.names = ['Date']

            valindex = valdata.index

            predictions = preds.copy()

            predictions.loc[:,'sales'] = inv_boxcox(predictions['sales'],0)

            valdata.loc[:,'Weekly_Sales'] = inv_boxcox(valdata['Weekly_Sales'],0)

            # following is done to modify number index (for non-periodic validation timseries) to datetime index.

            dtstr = str(predictions.index[0])

            words = dtstr.split("-")

            if (len(words)==1):

                predictions['Date'] = valdata.index

                predictions.set_index('Date',inplace=True)

#             print ("final pred\n",predictions)

            sales_join = pd.DataFrame(pd.merge(valdata['Weekly_Sales'],predictions['sales'],left_index=True,right_index=True))

            store_result = pd.DataFrame(predictions)

            store_result['Store'] = sid

            store_result['Dept'] = did

            store_result['Weekly_Sales'] = sales_join['Weekly_Sales']

            store_result['IsHoliday'] = valdata['IsHoliday']

            val_result = val_result.append(store_result,ignore_index=True)

            

        else:

            test_exo = ftest.loc[(ftest.Store==sid) & (ftest.Dept==did), exo_features]

            test_cnt = len(test_exo)

            # some of the store, deptid are present in train data but not in test data.

            if (test_cnt==0):

                continue

            start_row = trcnt

            end_row = trcnt+test_cnt-1

            predictions = sarimax_specific(trsales, trexo, test_exo, start_row, end_row)

            if (predictions is None):

                print ("Exception in train/test (sid, did)"+str(sid)+" "+str(did))

                continue

            predictions = inv_boxcox(predictions,0)

            preds = pd.DataFrame(predictions, columns=['Weekly_Sales'])

            preds = preds.reset_index()

            test_date = pd.DataFrame(test_exo.index.strftime('%Y-%m-%d'))

            test_date['Date'] = str(sid)+"_"+str(did)+"_"+test_date['Date']

            test_date.rename(columns={'Date':'Id'},inplace=True)

            test_date['Weekly_Sales'] = preds['Weekly_Sales']

#             print ("final result\n",test_date.head())

            fresult = fresult.append(test_date,ignore_index=True)     

            

    if (ftest is None):

        return pd.DataFrame(val_result)

    else:

        return fresult

            



def sarimax_specific(trsales, trexo, test_exo, start_row, end_row):

    non_seasonal_order=(0,1,1)

    myseasonal_order=(1,0,0,52)

    try:

        model = SARIMAX(trsales,order=non_seasonal_order, seasonal_order=myseasonal_order,

                        enforce_stationarity=False, enforce_invertibility=False,exogenous=trexo)

        model_fit = model.fit()

        predictions = model_fit.predict(start=start_row,end=end_row,exog=test_exo)

    except Exception as e:

        print (e)

        predictions = None

    return predictions

# training and validation of all the timeseries models

import time



warnings.filterwarnings(action='ignore')

t_start = time.time()

fresult = sarimax_driver(rtrain,0.7)

t_end = time.time()

print ("training and validation completed in "+str((t_end-t_start)/60)+" mins")

warnings.filterwarnings(action='once')
# there are still some null Weekly_sales rows(108 out of 0.4 million rows). so we are going to drop them.



val = len (fresult.loc[fresult.Weekly_Sales.isnull()])

print ("Null value rows count: "+str(val))

fresult.dropna(inplace=True)

val = WMAE(fresult,real=fresult['Weekly_Sales'],predicted=fresult['sales'])

print ("Validation score on all the timeseries models: "+ str(val))
# forecasting final test data

import time



warnings.filterwarnings(action='ignore')

t_start = time.time()

fresult = sarimax_driver(rtrain,1,rtest)

t_end = time.time()

print ("training and validation completed in "+str((t_end-t_start)/60)+" mins")

warnings.filterwarnings(action='once')
print (fresult.loc[fresult.Weekly_Sales.isnull()])

print (len(fresult))
# Evaluation Exception: Submission must have 115064 rows

# we have 114706 rows. We are not forecasting some of the sid_did pairs which has too less data (<10 rows, sometimes 0 rows of training data(10,99);)

# so lets add them with 0 value.



sample_sub = pd.DataFrame(pd.read_csv('/kaggle/working/sampleSubmission.csv')) #,parse_dates=[2],index_col=2,squeeze=True))

sample_sub_set = set(sample_sub['Id'])

sub_set = set(fresult["Id"])

diff_set = sample_sub_set.difference(sub_set)

diff_list = list (diff_set)

diff_df = pd.DataFrame()

diff_df['Id'] = diff_list

diff_df['Weekly_Sales'] = 0

diff_df
fresult = fresult.append(diff_df, ignore_index=True)
len(fresult)
fresult.to_csv("/kaggle/working/sarimax_basic.csv",index=False)