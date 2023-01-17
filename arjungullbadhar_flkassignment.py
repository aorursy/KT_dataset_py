%matplotlib inline



# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O ()

import pandas

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import lightgbm as lgb

import xgboost as xgb

import seaborn as sns



from fbprophet import Prophet



def ignore_warn(*args, **kwargs):

    pass





from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error

from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error

from sklearn.model_selection import KFold

from scipy import stats

from plotly.offline import init_notebook_mode, iplot

from plotly import graph_objs as go



import statsmodels.api as sm

# Initialize plotly

init_notebook_mode(connected=True)

def mean_absolute_percentage_error(y_true, y_pred): 

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100



pd.option_context("display.max_rows", 1000);

pd.option_context("display.max_columns", 1000);

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
subs = pd.read_csv(f'/kaggle/input/latest_asset_count.csv')

subs['date2'] = pd.Timestamp('2020-01-15')

subs['date'] = pd.to_datetime(subs['date'])

subs['days'] = (subs['date2'] - subs['date']).dt.days

subs.head()

subs.head()

subs[['date','sku_id',"count",'city_id']].groupby(['date','sku_id','city_id']).sum().sort_values('date',ascending=True).to_csv('abc.csv')
subs[['sku_id','count']].groupby(['sku_id']).sum().sort_values('count',ascending=False)
subs.dtypes
"""print(type(subs['date']))

subs['datetime'] = pd.to_datetime(subs.date + ' '+ '00:00:00', infer_datetime_format=True)

subs.set_index('datetime', inplace=True)

"""
print(subs['city_id'].unique())

print(subs['sku_id'].unique())



print(len(subs['city_id'].unique()))



print(len(subs['sku_id'].unique()))
print("The number of cities and products are ",end='')

print(len(subs['city_id'].unique()),end=" and ")



print(len(subs['sku_id'].unique()), end=" ")

print("respectively.")
subs.head()
print(subs.shape)
"""

subs['day_of_week'] = subs['date'].dt.weekday_name

print(subs.head(100))"""
subs['WEEKDAY'] = pandas.to_datetime(subs['date']).dt.dayofweek  # monday = 0, sunday = 6

subs['is_weekend'] = 0          # Initialize the column with default value of 0

subs.loc[subs['WEEKDAY'].isin([5, 6]), 'is_weekend'] = 1  # 5 and 6 correspond to Sat and Sun

del subs['WEEKDAY']

subs.drop(['date'], axis=1, inplace=True)

(subs.head(320))
#### Seasonality Check

# preparation: input should be float type

import matplotlib.pyplot as plt # basic plotting

subs['count'] = subs['count'] * 1.0



# store types

sales_2 = subs[subs.city_id == 2]['count'].sort_index(ascending = True)

sales_4 = subs[subs.city_id == 4]['count'].sort_index(ascending = True)

sales_6 = subs[subs.city_id == 6]['count'].sort_index(ascending = True)

sales_8 = subs[subs.city_id == 8]['count'].sort_index(ascending = True)

sales_10 = subs[subs.city_id == 10]['count'].sort_index(ascending = True)

sales_12 = subs[subs.city_id == 12]['count'].sort_index(ascending = True)

sales_14 = subs[subs.city_id == 14]['count'].sort_index(ascending = True)

sales_16 = subs[subs.city_id == 16]['count'].sort_index(ascending = True)



f, (ax1, ax2, ax3, ax4, ax5,ax6,ax7,ax8) = plt.subplots(8, figsize = (40, 40))

c = '#386B7F'



# store types

sales_2.plot(color = c, ax = ax1)

sales_4.plot(color = c, ax = ax2)

sales_6.plot(color = c, ax = ax3)

sales_8.plot(color = c, ax = ax4)

sales_10.plot(color = c, ax = ax5)

sales_12.plot(color = c, ax = ax6)

sales_14.plot(color = c, ax = ax7)

sales_16.plot(color = c, ax = ax8)



#All Stores have same trend... Weird Seems like the dataset is A Synthetic One..;
subs = subs.reset_index()

subs['date']=subs['datetime']

subs.head()

import re

def add_datepart(df, fldname, drop=True):



    """

    Parameters:

    -----------

    df: A pandas data frame. df gain several new columns.

    fldname: A string that is the name of the date column you wish to expand.

        If it is not a datetime64 series, it will be converted to one with pd.to_datetime.

    drop: If true then the original date column will be removed.

    """

    

    fld = df[fldname]

    fld_dtype = fld.dtype

    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):

        fld_dtype = np.datetime64



    if not np.issubdtype(fld_dtype, np.datetime64):

        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)

        

    targ_pre = re.sub('[Dd]ate$', '', fldname)

    

    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear','weekofyear',

            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']

    

    for n in attr: 

        df[targ_pre + n] = getattr(fld.dt, n.lower())

        

    if drop: 

        df.drop(fldname, axis=1, inplace=True)



add_datepart(subs,'date',False)

subs.nunique()
subs.dtypes
df_raw=subs

pivoted = pd.pivot_table(df_raw, values='count', columns='Year', index='Month')

pivoted.plot(figsize=(12,12));
pivoted = pd.pivot_table(df_raw, values='count' , columns='Year', index='Week')

pivoted.plot(figsize=(12,12));
ts=df_raw.groupby(["date"])["count"].sum()

ts.astype('float')



plt.figure(figsize=(16,6))

plt.plot(ts.rolling(window=12,center=False).mean(),label='Rolling Mean');

plt.plot(ts.rolling(window=12,center=False).std(),label='Rolling sd');

plt.legend();
pivoted = pd.pivot_table(df_raw, values='count' , columns='Month', index='Day')

pivoted.plot(figsize=(20,20));



#WE CAN SEE THE SALES RISING IN THE FIRST WEEK AND FALLING AT MONTH END
corr_all = subs.drop('datetime', axis = 1).corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr_all, dtype = np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize = (11, 9))



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr_all, mask = mask,

            square = True, linewidths = .5, ax = ax, cmap = "BuPu")      

plt.show()
"""temp_1 = subs.groupby(['Year','Month','sku_id'])['count'].mean().reset_index()

plt.figure(figsize=(20,150))

sns.swarmplot('sku_id', 'count', data=temp_1, hue = 'Month');

# Place legend to the right

plt.legend(bbox_to_anchor=(1, 1), loc=2);"""

temp_1 = df_raw.groupby(['Year','Month'])['count'].mean().reset_index()

plt.figure(figsize=(12,8));

sns.lmplot('Month','count',data = temp_1, hue='Year', fit_reg= False);



#THE PERIOD 10TH ,11TH MONTH OBSERVES CONSTANT SALE FOR ALL THE YEAR
temp_1 = df_raw.groupby(['Year'])['count'].mean().reset_index()

plt.figure(figsize=(12,8));

sns.factorplot('Year','count',data = temp_1, hue='Year', kind='point');
subs.head()

del subs['date']

del subs['Year']

del subs['Month'] 

del subs['Week'] 

del subs['Day']
subs.to_csv('dataset.csv',index=False)
subs.head()
cols = list(subs.columns.values) #Make a list of all of the columns in the df

cols.pop(cols.index('count'))

subs = subs[cols+['count']] 

subs.head()
#del subs['datetime']

subs['datetime'] = subs.datetime.values.astype(np.int64) // 10 ** 9

subs=subs.astype('float32')



print (subs)
train = np.expand_dims(subs.values[:, :-1], axis=2)

test = subs.values[:, -1:]

"""

X_test = np.expand_dims(subs.values[:, 1:], axis=2)

print(X_train.shape, Y_train.shape, X_test.shape)

"""
from sklearn import model_selection





"""from sklearn.model_selection import cross_val_score,StratifiedKFold,GridSearchCV



X_train,X_test,Y_train,Y_test=model_selection.train_test_split(train,test,test_size=0.25,random_state=0,shuffle=False)"""
"""print(X_train.shape, Y_train.shape, X_test.shape)"""

"""from keras.models import Sequential

from keras.layers import LSTM, Dense, Dropout

from keras import backend

 

def rmse(y_true, y_pred):

    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))



model = Sequential()

model.add(LSTM(units=64, input_shape=(13,1)))

model.add(Dropout(0.3))

model.add(Dense(1))



model.compile(loss='mse',

              optimizer='adam',

              metrics=[rmse])



model.summary()"""
"""from keras.callbacks import EarlyStopping



callbacks_list=[EarlyStopping(monitor="val_loss",min_delta=.001, patience=3,mode='auto')]



history = model.fit(X_train, Y_train, batch_size=4096, epochs=50,callbacks=callbacks_list)"""