import pandas as pd

import numpy as np

import sklearn as skl

import matplotlib.pyplot as plt

import seaborn as sns

import scipy as sc

from sklearn.metrics import roc_auc_score

import gc #importing garbage collector

import time

from scipy import signal







import warnings

warnings.filterwarnings('ignore')



%matplotlib inline  



SEED = 42

#Pandas - Displaying more rorws and columns

pd.set_option("display.max_rows", 500)

pd.set_option("display.max_columns", 500)
def reduce_mem_usage(df, verbose=True):

    numerics = ['int8','int16', 'int32', 'int64', 'float16', 'float32', 'float64']

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
timesteps = 14

startDay = 0
df_train = pd.read_csv('/kaggle/input/m5-forecasting-uncertainty/sales_train_validation.csv')

df_prices = pd.read_csv('/kaggle/input/m5-forecasting-uncertainty/sell_prices.csv')

df_days = pd.read_csv('/kaggle/input/m5-forecasting-uncertainty/calendar.csv')



df_train = reduce_mem_usage(df_train)

df_prices = reduce_mem_usage(df_prices)

df_days = reduce_mem_usage(df_days)
series_cols = df_train.columns[df_train.columns.str.contains("d_")].values

level_cols = df_train.columns[df_train.columns.str.contains("d_")==False].values
df_train.head(1)
sns.set_palette("colorblind")



fig, ax = plt.subplots(5,1,figsize=(20,28))

df_train[series_cols].sum().plot(ax=ax[0])

ax[0].set_title("Top-Level-1: Summed product sales of all stores and states")

ax[0].set_ylabel("Unit sales of all products");

df_train.groupby("state_id")[series_cols].sum().transpose().plot(ax=ax[1])

ax[1].set_title("Level-2: Summed product sales of all stores per state");

ax[1].set_ylabel("Unit sales of all products");

df_train.groupby("store_id")[series_cols].sum().transpose().plot(ax=ax[2])

ax[2].set_title("Level-3: Summed product sales per store")

ax[2].set_ylabel("Unit sales of all products");

df_train.groupby("cat_id")[series_cols].sum().transpose().plot(ax=ax[3])

ax[3].set_title("Level-4: Summed product sales per category")

ax[3].set_ylabel("Unit sales of all products");

df_train.groupby("dept_id")[series_cols].sum().transpose().plot(ax=ax[4])

ax[4].set_title("Level-4: Summed product sales per product department")

ax[4].set_ylabel("Unit sales of all products");
submission_sample =pd.read_csv('/kaggle/input/m5-forecasting-uncertainty/sample_submission.csv')

submission_sample.head(10)
# total number of series * number of quartiles * 2 (validation & evaluation)

42840*9*2
submission_sample.shape
temp_series = df_train

plt.figure(figsize=(12,8))

peak_days = []

x = np.count_nonzero(temp_series==0, axis=0)

peaks, _ = sc.signal.find_peaks(x, height=np.quantile(x,0.75), threshold=max(x)/25)

peak_d = temp_series.columns[peaks]

peak_days=peak_d

plt.plot(x)

plt.plot(peaks, x[peaks], "x", color='red')

    

plt.title('Number of Zero Sales per Day')

plt.ylabel('Number of Zero Sales')

plt.xlabel('Days')
peak_days
df_days[df_days['d'].isin(peak_days)]
peak_days_before=[]

peak_days_after=[]



for i, days in enumerate(peak_days):

    peak_days_before.append('d_'+str(np.int(peak_days[i][2:])-1))

    peak_days_after.append('d_'+str(np.int(peak_days[i][2:])+1))
df_train_no_outlier = df_train.copy().T[1:]

df_train_no_outlier.columns = df_train.T.iloc[0]



for x,y,z in zip(peak_days,peak_days_before,peak_days_after):

        df_train_no_outlier[df_train_no_outlier.index==x] = np.reshape([pd.concat([df_train_no_outlier[df_train_no_outlier.index==y],df_train_no_outlier[df_train_no_outlier.index==z]],axis=0).mean()],(1,30490))



df_train_no_outlier = df_train_no_outlier.T.reset_index()
df_train_no_outlier = pd.concat([df_train_no_outlier[level_cols],df_train_no_outlier[series_cols].apply(pd.to_numeric,downcast='float')],axis=1)

df_train_no_outlier = reduce_mem_usage(df_train_no_outlier)
df_train_no_outlier.info()
temp_series = df_train_no_outlier

plt.figure(figsize=(12,8))

x = np.count_nonzero(temp_series==0, axis=0)

plt.plot(x)

    

plt.title('Number of Zero Sales per Day')

plt.ylabel('Number of Zero Sales')

plt.xlabel('Days')

plt.ylim(0,30000)
del temp_series, peak_days_before, peak_days_after, peak_d, peak_days, peaks
del df_train
df_train_no_outlier.head()
series_cols = df_train_no_outlier.columns[df_train_no_outlier.columns.str.contains("d_")].values

level_cols = df_train_no_outlier.columns[df_train_no_outlier.columns.str.contains("d_")==False].values
Level1 = pd.DataFrame(df_train_no_outlier[series_cols].sum(),columns={'Total'}).T

Level2 = df_train_no_outlier.groupby("state_id")[series_cols].sum()

Level3 = df_train_no_outlier.groupby("store_id")[series_cols].sum()

Level4 = df_train_no_outlier.groupby("cat_id")[series_cols].sum()

Level5 = df_train_no_outlier.groupby("dept_id")[series_cols].sum()



Level6 = df_train_no_outlier.groupby(["state_id",'cat_id'])[series_cols].sum().reset_index()

Level6['index']=''

for row in range(len(Level6)):

    Level6['index'][row]=str(Level6['state_id'][row])+'_'+str(Level6['cat_id'][row])

Level6.set_index(Level6['index'],inplace=True)

Level6.drop(['state_id','cat_id','index'],axis=1,inplace=True)



Level7 = df_train_no_outlier.groupby(["state_id",'dept_id'])[series_cols].sum().reset_index()

Level7['index']=''

for row in range(len(Level7)):

    Level7['index'][row]=str(Level7['state_id'][row])+'_'+str(Level7['dept_id'][row])

Level7.set_index(Level7['index'],inplace=True)

Level7.drop(['state_id','dept_id','index'],axis=1,inplace=True)



Level8 = df_train_no_outlier.groupby(["store_id",'cat_id'])[series_cols].sum().reset_index()

Level8['index']=''

for row in range(len(Level8)):

    Level8['index'][row]=str(Level8['store_id'][row])+'_'+str(Level8['cat_id'][row])

Level8.set_index(Level8['index'],inplace=True)

Level8.drop(['store_id','cat_id','index'],axis=1,inplace=True)



Level9 = df_train_no_outlier.groupby(["store_id",'dept_id'])[series_cols].sum().reset_index()

Level9['index']=''

for row in range(len(Level9)):

    Level9['index'][row]=str(Level9['store_id'][row])+'_'+str(Level9['dept_id'][row])

Level9.set_index(Level9['index'],inplace=True)

Level9.drop(['store_id','dept_id','index'],axis=1,inplace=True)



Level10= df_train_no_outlier.groupby(["item_id"])[series_cols].sum()





Level11= df_train_no_outlier.groupby(["item_id",'state_id'])[series_cols].sum().reset_index()

Level11['index']=''

for row in range(len(Level11)):

    Level11['index'][row]=str(Level11['item_id'][row])+'_'+str(Level11['state_id'][row])

Level11.set_index(Level11['index'],inplace=True)

Level11.drop(['item_id','state_id','index'],axis=1,inplace=True)





Level12= df_train_no_outlier.copy()

Level12.set_index(Level12['id'],inplace=True, drop =True)

Level12.drop(level_cols,axis=1,inplace=True)



df=pd.concat([Level1,Level2,Level3,Level4,Level5,Level6,Level7,Level8,Level9,Level10,Level11,Level12])



del Level1,Level2,Level3,Level4,Level5,Level6,Level7,Level8,Level9,Level10,Level11,Level12
test = pd.concat([df.reset_index()['index'],submission_sample.reset_index().id[:42840]],axis=1)

test

test['index'].replace('_validation','',regex=True,inplace=True)
test['proof'] = ''

for row in range(len(test)):

    if test['index'][row] in test['id'][row]:

        test['proof'][row]=True

test[test['proof']==False]
del test
df_days["date"] = pd.to_datetime(df_days['date'])

df_days.set_index('date', inplace=True)



df_days['is_event_day'] = [1 if x ==False else 0 for x in df_days['event_name_1'].isnull()] 

df_days['is_event_day'] = df_days['is_event_day'].astype(np.int8)



day_before_event = df_days[df_days['is_event_day']==1].index.shift(-1,freq='D')

df_days['is_event_day_before'] = 0

df_days['is_event_day_before'][df_days.index.isin(day_before_event)] = 1

df_days['is_event_day_before'] = df_days['is_event_day_before'].astype(np.int8)



del day_before_event



daysBeforeEventTest = df_days['is_event_day_before'][1913:1941]

daysBeforeEvent = df_days['is_event_day_before'][startDay:1913]

daysBeforeEvent.index = df_train_no_outlier.index[startDay:1913]
df_final = pd.concat([df.T.reset_index(drop=True), daysBeforeEvent.reset_index(drop=True)], axis = 1)

df_final = df_final[startDay:]
df_days
features = df_days[['is_event_day_before','wday','snap_CA','snap_TX','snap_WI']]

features.head()
# adding 'id' column as well as 'cat_id', 'dept_id' and 'state_id', then changing the type to 'categorical'

df_prices.loc[:, "id"] = df_prices.loc[:, "item_id"] + "_" + df_prices.loc[:, "store_id"] + "_validation"

df_prices['state_id'] = df_prices['store_id'].str.split('_',expand=True)[0]

df_prices = pd.concat([df_prices, df_prices["item_id"].str.split("_", expand=True)], axis=1)

df_prices = df_prices.rename(columns={0:"cat_id", 1:"dept_id"})

df_prices[["store_id", "item_id", "cat_id", "dept_id", 'state_id']] = df_prices[["store_id","item_id", "cat_id", "dept_id", 'state_id']].astype("category")

df_prices = df_prices.drop(columns=2)
price_features = pd.DataFrame(df_prices.groupby(['wm_yr_wk','store_id','cat_id'])['sell_price'].mean().reset_index())

price_features['sell_price'] = price_features['sell_price'].astype('float32')
price_features['store_cat'] = 0



for row in range(len(price_features)):

     price_features['store_cat'][row]=str(price_features['store_id'][row])+'_'+str(price_features['cat_id'][row])
price_features= price_features.pivot(index='store_cat',columns='wm_yr_wk',values='sell_price').T

price_features.head()
features = df_days[['wm_yr_wk','is_event_day_before','wday','snap_CA','snap_TX','snap_WI']]

features.head()

features = pd.merge(features.reset_index(),price_features,how='left', left_on='wm_yr_wk', right_on='wm_yr_wk').set_index('date')

features.drop('wm_yr_wk', axis=1, inplace=True)

features.head()
features_test = features.iloc[1913:1941,:]

features_train = features.iloc[startDay:1913,:]

df_final_more = pd.concat([df.T.reset_index(drop=True), features_train.reset_index(drop=True)], axis = 1)

df_final = df_final_more.copy()
del df_final_more, features
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

dt_scaled = sc.fit_transform(df_final)
gc.collect()
X_train = []

y_train = []

for i in range(timesteps, 1913 - startDay):

    X_train.append(dt_scaled[i-timesteps:i])

    y_train.append(dt_scaled[i][0:42840]) 

    

X_train = np.array(X_train)

y_train = np.array(y_train)

print('Shape of X_train :'+str(X_train.shape))

print('Shape of y_train :'+str(y_train.shape))
inputs = df_final[-timesteps:]

inputs = sc.transform(inputs)
%who
del df_train_no_outlier, df_prices, df_days, df, df_final, dt_scaled, price_features
gc.collect()
def tilted_loss(q, y, f):

    e = (y - f)

    return keras.backend.mean(keras.backend.maximum(q * e, (q - 1) * e), 

                              axis=-1)
QUANTILES = [0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995]
EPOCHS = 32 # going through the dataset 32 times

BATCH_SIZE = 32 # with each training step the model sees 32 examples
# Importing the Keras libraries and packages

import tensorflow_probability as tfp

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout

import tensorflow as tf

import keras



def run_model(X_train, y_train, q):



    model = Sequential()



    # Adding the first LSTM layer and some Dropout regularisation

    layer_1_units=40

    model.add(LSTM(units = layer_1_units, return_sequences = True, input_shape = (X_train.shape[1], X_train.shape[2])))

    model.add(Dropout(0.2))



    # Adding a second LSTM layer and some Dropout regularisation

    layer_2_units=300

    model.add(LSTM(units = layer_2_units, return_sequences = True))

    model.add(Dropout(0.2))



    # Adding a third LSTM layer and some Dropout regularisation

    layer_3_units=300

    model.add(LSTM(units = layer_3_units))

    model.add(Dropout(0.2))



    # Adding the output layer

    model.add(Dense(units = y_train.shape[1]))



    # Compiling the RNN

    model.compile(optimizer = 'adam',loss=lambda y, f: tilted_loss(q, y, f))

    

    # To follow at which quantile we are predicting right now  

    print('Running the model for Quantil: '+str(q)+':')



    # Fitting the RNN to the Training set

    fit = model.fit(X_train, y_train, epochs = EPOCHS, batch_size = BATCH_SIZE, verbose=2)

    

    X_test = []

    X_test.append(inputs[0:timesteps])

    X_test = np.array(X_test)

    prediction = []

     

    for j in range(timesteps,timesteps + 28):

        predicted_volume = model.predict(X_test[0,j - timesteps:j].reshape(1, timesteps, 42875)) #incl. features

        testInput = np.column_stack((np.array(predicted_volume), np.array(features_test.iloc[j-timesteps,:]).reshape(1,35))) #here no of features is 5

        X_test = np.append(X_test, testInput).reshape(1,j + 1,42875) #incl. features

        predicted_volume = sc.inverse_transform(testInput)[:,0:42840] #without features

        prediction.append(predicted_volume)

    

    prediction = pd.DataFrame(data=np.array(prediction).reshape(28,42840)).T

    return prediction
# We run the model for all the quantiles mentioned above. 

# Combining all quantile predictions one after another to a large dataset.

predictions = pd.concat(

    [run_model(X_train, y_train, q) 

     for q in QUANTILES]) 
gc.collect()
predictions.shape
predictions.shape[0]*2
predictions.to_pickle('Uncertainty_Predictions.pkl')
submission = pd.concat((predictions, predictions), ignore_index=True)

idColumn = submission_sample[["id"]]    

submission[["id"]] = idColumn  



#re-arranging collumns

cols = list(submission.columns)

cols = cols[-1:] + cols[:-1]

submission = submission[cols]

#

colsname = ["id"] + [f"F{i}" for i in range (1,29)]

submission.columns = colsname



submission.to_csv("submission.csv", index=False)
temp_series = submission[171360:171360+42840]



border = [1,3,10,3,7,9,21,30,70,3049,9147,30490]

sumi = 0

levels =[]

for i in border:

    sumi += i

    levels.append(pd.DataFrame(temp_series[sumi-i:sumi]))





for i,level in enumerate(levels):

    levels[i] = levels[i].sum()
plt.figure(figsize=(20, 8))



for i in range(12):

    plt.plot(levels[i][1:],label='level'+str(i+1))



plt.legend()
fig,ax = plt.subplots(figsize=(20, 8))



for i in range(6):

    ax = ax.twinx()

    ax.plot(levels[i][1:],label='level'+str(i+1))

    plt.yticks([])