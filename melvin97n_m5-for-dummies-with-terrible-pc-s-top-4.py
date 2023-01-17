#importing libraries for the analysis:

import pandas as pd

from datetime import datetime

import matplotlib.pyplot as plt

import seaborn as sns

from statsmodels.tsa.seasonal import seasonal_decompose

from matplotlib import dates

# Load specific forecasting tools

from statsmodels.tsa.statespace.sarimax import SARIMAX

!pip install --upgrade numpy==1.18.3 

import numpy as np

!pip install pmdarima

from pmdarima import auto_arima                              # for determining ARIMA orders
# Converting the csv files to pandas dataframe:

calender=pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv',parse_dates=True)

sales=pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_evaluation.csv',parse_dates=True)

prices=pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')

calender['date']=pd.to_datetime(calender['date'])

sample_submission=pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv')
# get the dimensions of the first dataframe 'calender' and show the first five rows:

print(calender.shape)

calender.head()
# Information on the different columns of the dataframe:

calender.info()
# get the dimensions of the first dataframe 'sales' and show the first five rows:

print(sales.shape)

sales.head(5)
# Information on the different columns of the dataframe:

sales.info()
# get the dimensions of the first dataframe 'prices' and show the first five rows:

print(prices.shape)

prices.head()
# Information on the different columns of the dataframe:

prices.info()
#Converting dtypes to 'categorical' and filling nans with -1( we use -1 to further reduce the dataset size as compared to nans) for all the 3 dataframes:

calender=calender.fillna(-1)

calender[['event_type_1','event_type_2','event_name_1','event_name_2']]=calender[['event_type_1','event_type_2','event_name_1','event_name_2']].astype(('category'))



sales[['id','item_id','cat_id','store_id','state_id']]=sales[['id','item_id','cat_id','store_id','state_id']].astype('category')



# For prices column we combine the item_id and store_id to form the id of the data which can later be joined with sales dataframe:

prices['id']=prices['item_id']+'_'+prices['store_id']+'_evaluation'



prices[['id','store_id','item_id']]=prices[['id','store_id','item_id']].astype('category')

# We also drop store_id and item_id as they no longer play any role in the dataset and all the information is stored in 'id'.

prices.drop(['store_id','item_id'],axis=1,inplace=True)

# We also drop dept_id from sales as we will note be using the column:

sales.drop('dept_id',axis=1,inplace=True)

# This very convinient piece of code is commonly found on kaggle competitions which performs the above tasks for all the rows:

def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df

    
# Applying the above function to the prices dataframe and calander dataframe:

# We apply the function to sales dataframe after applying melt to sales:

prices=reduce_mem_usage(prices)

calender=reduce_mem_usage(calender)
# We use pd.melt to do the task above, which essentially bring the table to the format given below:

sales=pd.melt(sales,id_vars=['id','item_id','cat_id','store_id','state_id'])

sales.head()
sales=reduce_mem_usage(sales)
# Here we merge all three dataframes:

df=pd.merge(pd.merge(calender[['date','d','wm_yr_wk',

       'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2',

       'snap_CA', 'snap_TX', 'snap_WI']],sales,left_on='d',right_on='variable',how='inner'),prices,left_on=['id','wm_yr_wk'],right_on=['id','wm_yr_wk'],how='inner')



# We get rid of the columns on which the dataframe was joined on as we already have the date column instead:

df.drop(['d','variable','wm_yr_wk'],axis=1,inplace=True)

# Rearranging the columns to our convinience:

cols=['date','id', 'item_id', 'cat_id', 'store_id', 'state_id','sell_price','event_name_1', 'event_type_1', 'event_name_2','event_type_2', 'snap_CA', 'snap_TX', 'snap_WI','value']

df=df[cols]

df.tail()
# Given below is information on the various columns of the dataframe:

df.info()
df.tail()
df[['date','value']].groupby('date').agg({'value':'sum'}).plot(figsize=(20,8),grid=True);
df.loc[(df.date>'2015-01-01')&(df.date<'2016-01-01')][['date','value']].groupby('date').agg({'value':'sum'}).plot(figsize=(20,8),grid=True);
ax=df.loc[(df.date>'2014-08-01')&(df.date<'2014-09-01')][['date','value']].groupby('date').agg({'value':'sum'}).plot(figsize=(20,8))

ax.xaxis.set_minor_locator(dates.DayLocator())

ax.xaxis.set_minor_formatter(dates.DateFormatter("%a-%B-%d"))

ax.tick_params(which='minor', rotation=45)

ax.grid(b=True, which='minor')
storewise=df[['date','store_id','value']].groupby(['date','store_id']).agg({'value':'sum'})

storewise.reset_index(inplace=True)

storewise.pivot(index="date", columns="store_id", values="value").rolling(window=90).mean().plot(figsize=(20,8),grid=True,title='Sum of sales by store');
category_wise=df[['date','cat_id','value']].groupby(['date','cat_id']).agg({'value':'sum'})

category_wise.reset_index(inplace=True)

category_wise.pivot(index="date", columns="cat_id", values="value").rolling(window=90).mean().plot(figsize=(20,8),grid=True,title='Sum of sales by category');
statewise=df[['date','state_id','value']].groupby(['date','state_id']).agg({'value':'sum'})

statewise.reset_index(inplace=True)

statewise.pivot(index="date", columns="state_id", values="value").rolling(window=90).mean().plot(figsize=(20,8),grid=True,title='Sum of sales by state');
item1=df[['item_id','sell_price','value']].loc[df['item_id']=='HOBBIES_1_008']

sns.barplot(x='sell_price',y='value',data=item1).set_title('Item1')

sns.set(rc={'figure.figsize':(10,5)})

plt.show()

item1= df[['cat_id','sell_price','value']].loc[df['cat_id']=='FOODS']

sns.scatterplot(x='sell_price',y='value',data=item1).set_title('Effect of price on sales for food')

sns.set(rc={'figure.figsize':(10,5)})

plt.show()
ax=sns.barplot(x='event_name_1',y='value',data=df[['event_name_1','value']].groupby('event_name_1').agg({'value':'mean'}).sort_values(['value']).reset_index())

ax.tick_params(which='both',rotation=90)

sns.set(rc={'figure.figsize':(20,8)})
fig, ax =plt.subplots(1,3)

sns.barplot(x='snap_CA',y='value',data=df[['snap_CA','value']].groupby('snap_CA').agg({'value':'mean'}).sort_values(['value']).reset_index(),ax=ax[0])

sns.set(rc={'figure.figsize':(10,6)})

sns.barplot(x='snap_TX',y='value',data=df[['snap_TX','value']].groupby('snap_TX').agg({'value':'mean'}).sort_values(['value']).reset_index(),ax=ax[1])

sns.barplot(x='snap_WI',y='value',data=df[['snap_WI','value']].groupby('snap_WI').agg({'value':'mean'}).sort_values(['value']).reset_index(),ax=ax[2])

plt.show()
df['snap']=np.where(df['state_id']=='CA',df['snap_CA'] ,np.where( df['state_id']=='TX',df['snap_TX'],np.where(df['state_id']=='WI',df['snap_WI'],0 )))
item1=df[['date', 'id', 'cat_id', 'sell_price','event_name_1',  'event_name_2','snap_CA', 'snap_TX', 'snap_WI','snap', 'value']].loc[df['id']=='HOBBIES_1_001_CA_1_evaluation']
train=item1[:-28]

test=item1.iloc[-28:]
len(test)
item1.set_index('date')

auto_arima(train['value'],seasonal=True,m=7,start_Q=0,start_P=0).summary()
model = SARIMAX(train['value'],exog=train[['sell_price','snap']].astype('float'),order=(0,1,1))

results = model.fit()

results.summary()
exog=test[['snap', 'sell_price']].astype(float)

predictions=results.predict(start=len(train),end=len(train)+len(test)-1,exog=exog)
predictions=predictions.to_numpy()
test=test.copy()



test['predictions']=predictions
test.reset_index(drop=True,inplace=True)
test[['value','predictions']].plot()
from statsmodels.tools.eval_measures import mse,rmse

RMSE= rmse(test['value'],test['predictions'])

print("RMSE:",RMSE)
#Importing required libraries:

import pandas as pd

from fbprophet import Prophet

from tqdm.notebook import tqdm
# Forming the holidays dataframe:

holidays_df=df[['date','event_name_1']].loc[df['event_name_1']!=-1]

holidays_df.drop_duplicates(inplace=True)
holidays_df.columns=['ds','holiday']

holidays_df
# Making the columns in the required format:

cols=['ds','y','sell_price','snap']

id=sample_submission['id']
# Training and fitting the data:

item=train[['date','value','sell_price','snap']]

item.columns=cols  

m = Prophet(weekly_seasonality=True,holidays=holidays_df)

m.add_regressor('sell_price')

m.add_regressor('snap')

m.fit(item[-365:])
#  Forming the dataframe for our forecast:

future = m.make_future_dataframe(periods=28)

future['sell_price']=item1['sell_price'][-393:].to_numpy()

future['snap']=item1['snap'][-393:].to_numpy()
# Forecasting the data:

forecast = m.predict(future)[-28:]
# Putting the forecasted data with the actual values to test:

preds=forecast['yhat']

preds=preds.to_numpy()

test['yhat']=preds
# Plotting the data:

m.plot(m.predict(future));
test[['yhat','value']].plot();
RMSE = rmse(test['yhat'],test['value'])

print("RMSE:",RMSE) 
# Forming the necessary dataframes for our final model:

cols=['ds','y','sell_price','snap_TX']

id=sample_submission['id']

# Forming the submission dataframe:

submission=pd.DataFrame(index=('F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10',

       'F11', 'F12', 'F13', 'F14', 'F15', 'F16', 'F17', 'F18', 'F19', 'F20',

       'F21', 'F22', 'F23', 'F24', 'F25', 'F26', 'F27', 'F28'))

future_data=pd.merge(prices[['id','sell_price','wm_yr_wk']],calender[['date','wm_yr_wk']],left_on='wm_yr_wk',right_on='wm_yr_wk')
# To find where 'ID' for California ends and Texas begins:

print(id[id=='HOBBIES_1_001_TX_1_evaluation'].index)

print(id[id=='HOBBIES_1_001_WI_1_evaluation'].index)
def predict(i) :

   item=df[['date','value','sell_price','snap_TX']].loc[df['id']==i]

   future_id=future_data[['sell_price','date']].loc[future_data['id']==i].sort_values('date')

   item.columns=cols  

   m = Prophet(yearly_seasonality=False,daily_seasonality=False,holidays=holidays_df)

   m.add_regressor('sell_price')

   m.add_regressor('snap_TX')

   m.fit(item[-365:])

   future = m.make_future_dataframe(periods=28)[-28:]

   future['sell_price']=future_id['sell_price'][-28:].to_numpy()

   future['snap_TX']=calender['snap_TX'][-28:].to_numpy()

   forecast = m.predict(future)[['yhat']]

   submission[i]=forecast.to_numpy()

   if n%100==0:

     submission.to_csv('submission_TX.csv')

for i,n in zip(tqdm(id[47590:47592]),range(0,2)):

  predict(i)
submission.tail()
# Transposing them from being horizontal to vertical:

TX1=TX1.transpose()

TX2=TX2.transpose()

WI1=WI1.transpose()

WI2=WI2.transpose()

WI3=WI3.transpose()

CA1=CA1.transpose()

CA2=CA2.transpose()

CA3=CA3.transpose()

CA4=CA4.transpose()

CA5=CA5.transpose()

CA6=CA6.transpose()

CA7=CA7.transpose()

CA8=CA8.transpose()
#Combining all of them into one dataframe:

submission=CA1.append([CA2,CA3,CA4,CA5,CA6,CA7,CA8,TX1,TX2,WI1,WI2,WI3])
# Formatting to specifications for kaggle submission:

submission.reset_index(inplace=True)

submission.columns=[sample_submission.columns]

submission.to_csv('submission.csv')

submission=pd.read_csv('/content/submission.csv')

final=sample_submission[:30490].append(submission)

final.drop('Unnamed: 0',inplace=True,axis=1)

final.reset_index(drop=True,inplace=True)

final.to_csv('final_submission.csv')