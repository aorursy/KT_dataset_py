# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #charting

from scipy.stats import mode #statistics for slope

from sklearn.metrics import mean_squared_error #error metric to optimise when we build a model

from math import sqrt #Other math functions

import plotly.express as px #alternative charting function

import lightgbm as lgb #popular model choice

import seaborn as sns #alternative charting function

import gc



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#change display when printing .head for example the default settings only displays limited number of columns

pd.set_option('display.max_columns', 500)
#The data we will be using is a foreign exhange rates dataset kindly provided to Kaggle. 

forex_df = pd.read_csv('/kaggle/input/foreign-exchange-rates-per-dollar-20002019/Foreign_Exchange_Rates.csv',engine = 'python')

#let's go ahead and preview the top or bottom n rows

forex_df.tail(6)
#you can see that the currencies are 'objects' which effectively means they are strings. We will need to convert later on to numeric to enable calculations.

forex_df.info()
#What is the number of rows, and columns in this dataset?

forex_df.shape
#create a list of all the currency columns in the dataset

currency_list = ['AUSTRALIA - AUSTRALIAN DOLLAR/US$','EURO AREA - EURO/US$','NEW ZEALAND - NEW ZELAND DOLLAR/US$','UNITED KINGDOM - UNITED KINGDOM POUND/US$','BRAZIL - REAL/US$','CANADA - CANADIAN DOLLAR/US$','CHINA - YUAN/US$','HONG KONG - HONG KONG DOLLAR/US$','INDIA - INDIAN RUPEE/US$','KOREA - WON/US$','MEXICO - MEXICAN PESO/US$','SOUTH AFRICA - RAND/US$','SINGAPORE - SINGAPORE DOLLAR/US$','DENMARK - DANISH KRONE/US$','JAPAN - YEN/US$','MALAYSIA - RINGGIT/US$','NORWAY - NORWEGIAN KRONE/US$','SWEDEN - KRONA/US$','SRI LANKA - SRI LANKAN RUPEE/US$','SWITZERLAND - FRANC/US$','TAIWAN - NEW TAIWAN DOLLAR/US$','THAILAND - BAHT/US$']

#cleanse data

for c in currency_list:

    #ffill simply takes the previous row and applies it to the next row. We have conditioned this to only be applied to non numeric data.

    forex_df[c] = forex_df[c].where(~forex_df[c].str.isalpha()).ffill()

    #we then want to convert the currency columns into numeric so that we can apply functions to it.

    forex_df[c] = pd.to_numeric(forex_df[c], errors='coerce') 
#Let's check that this actually did what we intended.

forex_df.tail()
#let's check that the columns are now numeric, yep that worked!

forex_df.info()
#generate features



# time features

forex_df['date'] = pd.to_datetime(forex_df['Time Serie'])

forex_df['year'] = forex_df['date'].dt.year

forex_df['month'] = forex_df['date'].dt.month

forex_df['week'] = forex_df['date'].dt.week

forex_df['day'] = forex_df['date'].dt.day

forex_df['dayofweek'] = forex_df['date'].dt.dayofweek



# lag features

forex_df['lag_t1'] = forex_df['AUSTRALIA - AUSTRALIAN DOLLAR/US$'].transform(lambda x: x.shift(1))

forex_df['lag_t3'] = forex_df['AUSTRALIA - AUSTRALIAN DOLLAR/US$'].transform(lambda x: x.shift(3))

forex_df['lag_t7'] = forex_df['AUSTRALIA - AUSTRALIAN DOLLAR/US$'].transform(lambda x: x.shift(7))

# lag other country features

for c in [x for x in currency_list if x != "AUSTRALIA - AUSTRALIAN DOLLAR/US$"]:

    forex_df['lag_t1_%s' % c] = forex_df[c].transform(lambda x: x.shift(1))
# ratio lag other country features

for c in [x for x in currency_list if x != "AUSTRALIA - AUSTRALIAN DOLLAR/US$"]:

    forex_df['lag_t1_ratio_%s' % c] = forex_df['lag_t1']  / forex_df['lag_t1_' + c] 
forex_df.tail()
#rolling features

#mean

forex_df['rolling_mean_t1_t7'] = forex_df['lag_t1'].rolling(7,min_periods=1).mean()

forex_df['rolling_mean_t1_t14'] = forex_df['lag_t1'].rolling(14,min_periods=1).mean()

forex_df['rolling_mean_t1_t28'] = forex_df['lag_t1'].rolling(28,min_periods=1).mean()

forex_df['rolling_mean_t1_t90'] = forex_df['lag_t1'].rolling(90,min_periods=1).mean()

forex_df['rolling_mean_t1_t180'] = forex_df['lag_t1'].rolling(180,min_periods=1).mean()

forex_df['rolling_mean_t1_t360'] = forex_df['lag_t1'].rolling(360,min_periods=1).mean()



#max

forex_df['rolling_max_t1_t7'] = forex_df['lag_t1'].rolling(7,min_periods=1).max()

forex_df['rolling_max_t1_t14'] = forex_df['lag_t1'].rolling(14,min_periods=1).max()

forex_df['rolling_max_t1_t28'] = forex_df['lag_t1'].rolling(28,min_periods=1).max()

forex_df['rolling_max_t1_t90'] = forex_df['lag_t1'].rolling(90,min_periods=1).max()

forex_df['rolling_max_t1_t180'] = forex_df['lag_t1'].rolling(180,min_periods=1).max()

forex_df['rolling_max_t1_t360'] = forex_df['lag_t1'].rolling(360,min_periods=1).max()



#min

forex_df['rolling_min_t1_t7'] = forex_df['lag_t1'].rolling(7,min_periods=1).min()

forex_df['rolling_min_t1_t14'] = forex_df['lag_t1'].rolling(14,min_periods=1).min()

forex_df['rolling_min_t1_t28'] = forex_df['lag_t1'].rolling(28,min_periods=1).min()

forex_df['rolling_min_t1_t90'] = forex_df['lag_t1'].rolling(90,min_periods=1).min()

forex_df['rolling_min_t1_t180'] = forex_df['lag_t1'].rolling(180,min_periods=1).min()

forex_df['rolling_min_t1_t360'] = forex_df['lag_t1'].rolling(360,min_periods=1).min()



#standard deviation

forex_df['rolling_std_t1_t7'] = forex_df['lag_t1'].rolling(7,min_periods=1).std()

forex_df['rolling_std_t1_t14'] = forex_df['lag_t1'].rolling(14,min_periods=1).std()

forex_df['rolling_std_t1_t28'] = forex_df['lag_t1'].rolling(28,min_periods=1).std()

forex_df['rolling_std_t1_t90'] = forex_df['lag_t1'].rolling(90,min_periods=1).std()

forex_df['rolling_std_t1_t180'] = forex_df['lag_t1'].rolling(180,min_periods=1).std()

forex_df['rolling_std_t1_t360'] = forex_df['lag_t1'].rolling(360,min_periods=1).std()



#median

forex_df['rolling_med_t1_t7'] = forex_df['lag_t1'].rolling(7,min_periods=1).median()

forex_df['rolling_med_t1_t14'] = forex_df['lag_t1'].rolling(14,min_periods=1).median()

forex_df['rolling_med_t1_t28'] = forex_df['lag_t1'].rolling(28,min_periods=1).median()

forex_df['rolling_med_t1_t90'] = forex_df['lag_t1'].rolling(90,min_periods=1).median()

forex_df['rolling_med_t1_t180'] = forex_df['lag_t1'].rolling(180,min_periods=1).median()

forex_df['rolling_med_t1_t360'] = forex_df['lag_t1'].rolling(360,min_periods=1).median()
# exponential moving averages

forex_df['rolling_ema_t1_t7'] = forex_df['lag_t1'].ewm(span=7,adjust=False).mean()

forex_df['rolling_ema_t1_t14'] = forex_df['lag_t1'].ewm(span=14,adjust=False).mean()

forex_df['rolling_ema_t1_t28'] = forex_df['lag_t1'].ewm(span=28,adjust=False).mean()

forex_df['rolling_ema_t1_t90'] = forex_df['lag_t1'].ewm(span=90,adjust=False).mean()

forex_df['rolling_ema_t1_t180'] = forex_df['lag_t1'].ewm(span=180,adjust=False).mean()

forex_df['rolling_ema_t1_t360'] = forex_df['lag_t1'].ewm(span=360,adjust=False).mean()
#Take a quick look at the data over time now that we have some features to compare against:

# This is a relatively easy method to plot multiple values on a line chart plus it allows you to dynamically interact with the chart

df_long=pd.melt(forex_df, id_vars=['date'], value_vars=['AUSTRALIA - AUSTRALIAN DOLLAR/US$', 'rolling_ema_t1_t7', 'rolling_mean_t1_t7', 'rolling_ema_t1_t360', 'rolling_med_t1_t360'])



# plotly 

fig = px.line(df_long, x='date', y='value', color='variable')



# Show plot 

fig.show()
#round the value to 0 decimals

forex_df['lag_t1_round_0'] = forex_df['lag_t1'].round(0)

forex_df['lag_t3_round_0'] = forex_df['lag_t3'].round(0)

forex_df['lag_t7_round_0'] = forex_df['lag_t7'].round(0)



#get the decimal place

forex_df['lag_t1_dec'] = forex_df['lag_t1'] - forex_df['lag_t1_round_0']

forex_df['lag_t3_dec'] = forex_df['lag_t3'] - forex_df['lag_t3_round_0']

forex_df['lag_t7_dec'] = forex_df['lag_t7'] - forex_df['lag_t7_round_0']



#round the value to 1 decimals, as the rounded value to 0 decimals is nearly always 1 in the case of AUD/USD

forex_df['lag_t1_round_1'] = forex_df['lag_t1'].round(1)

forex_df['lag_t3_round_1'] = forex_df['lag_t3'].round(1)

forex_df['lag_t7_round_1'] = forex_df['lag_t7'].round(1)
#rolling mode of rounded figure

forex_df['lag_t1_mode_7'] = forex_df['lag_t1_round_1'].rolling(window=7,min_periods=1).apply(lambda x: mode(x)[0])

forex_df['lag_t1_mode_14'] = forex_df['lag_t1_round_1'].rolling(window=14,min_periods=1).apply(lambda x: mode(x)[0])

forex_df['lag_t1_mode_28'] = forex_df['lag_t1_round_1'].rolling(window=28,min_periods=1).apply(lambda x: mode(x)[0])

forex_df['lag_t1_mode_90'] = forex_df['lag_t1_round_1'].rolling(window=90,min_periods=1).apply(lambda x: mode(x)[0])

forex_df['lag_t1_mode_180'] = forex_df['lag_t1_round_1'].rolling(window=180,min_periods=1).apply(lambda x: mode(x)[0])

forex_df['lag_t1_mode_360'] = forex_df['lag_t1_round_1'].rolling(window=360,min_periods=1).apply(lambda x: mode(x)[0])
#frequency of mode

#ranges

forex_df['rolling_range_t1_t7'] = forex_df['rolling_max_t1_t7'] - forex_df['rolling_min_t1_t7']

forex_df['rolling_range_t1_t14'] = forex_df['rolling_max_t1_t14'] - forex_df['rolling_min_t1_t14']

forex_df['rolling_range_t1_t28'] = forex_df['rolling_max_t1_t28'] - forex_df['rolling_min_t1_t28']

forex_df['rolling_range_t1_t90'] = forex_df['rolling_max_t1_t90'] - forex_df['rolling_min_t1_t90']

forex_df['rolling_range_t1_t180'] = forex_df['rolling_max_t1_t180'] - forex_df['rolling_min_t1_t180']

forex_df['rolling_range_t1_t360'] = forex_df['rolling_max_t1_t360'] - forex_df['rolling_min_t1_t360']
#coefficient of variation - the ratio of standard deviation to mean

forex_df['rolling_coefvar_t1_t7'] =  forex_df['rolling_std_t1_t7'] / forex_df['rolling_mean_t1_t7']

forex_df['rolling_coefvar_t1_t14'] = forex_df['rolling_std_t1_t14'] / forex_df['rolling_mean_t1_t14']

forex_df['rolling_coefvar_t1_t28'] = forex_df['rolling_std_t1_t28'] / forex_df['rolling_mean_t1_t28']

forex_df['rolling_coefvar_t1_t90'] = forex_df['rolling_std_t1_t90'] / forex_df['rolling_mean_t1_t90']

forex_df['rolling_coefvar_t1_t180'] = forex_df['rolling_std_t1_t180'] / forex_df['rolling_mean_t1_t180']

forex_df['rolling_coefvar_t1_t360'] = forex_df['rolling_std_t1_t360'] / forex_df['rolling_mean_t1_t360']
#ratio of change to standard deviation

#I like this because if the currency is normally volatile (high std dev), then a change in the rolling mean may be normal. 

#On the other hand if the currency is not normally volatile (low std dev), then it adds weight to any changes observed

forex_df['rolling_meanstd_t1_t14'] = (forex_df['rolling_mean_t1_t7'] - forex_df['rolling_mean_t1_t14']) / forex_df['rolling_std_t1_t14']

forex_df['rolling_meanstd_t1_t28'] = (forex_df['rolling_mean_t1_t7'] - forex_df['rolling_mean_t1_t28']) / forex_df['rolling_std_t1_t28']

forex_df['rolling_meanstd_t1_t90'] = (forex_df['rolling_mean_t1_t7'] - forex_df['rolling_mean_t1_t90']) / forex_df['rolling_std_t1_t90']

forex_df['rolling_meanstd_t1_t180'] = (forex_df['rolling_mean_t1_t7'] - forex_df['rolling_mean_t1_t180']) / forex_df['rolling_std_t1_t180']

forex_df['rolling_meanstd_t1_t360'] = (forex_df['rolling_mean_t1_t7'] - forex_df['rolling_mean_t1_t360']) / forex_df['rolling_std_t1_t360']
#cardinality

forex_df['lag_t1_card_180'] = forex_df['lag_t1_round_1'].rolling(window=180,min_periods=1).apply(lambda x: np.unique(x).shape[0])

forex_df['lag_t1_card_360'] = forex_df['lag_t1_round_1'].rolling(window=360,min_periods=1).apply(lambda x: np.unique(x).shape[0])
#moving average crossover trends, 1 = positive, 0 = negative

forex_df['lag_t1_trend_7'] = np.where(forex_df['lag_t1'] >= forex_df['rolling_ema_t1_t7'],1,0)

forex_df['lag_t1_trend_14'] = np.where(forex_df['rolling_ema_t1_t7'] >= forex_df['rolling_ema_t1_t14'],1,0)

forex_df['lag_t1_trend_28'] = np.where(forex_df['rolling_ema_t1_t7'] >= forex_df['rolling_ema_t1_t28'],1,0)

forex_df['lag_t1_trend_90'] = np.where(forex_df['rolling_ema_t1_t7'] >= forex_df['rolling_ema_t1_t90'],1,0)

forex_df['lag_t1_trend_180'] = np.where(forex_df['rolling_ema_t1_t7'] >= forex_df['rolling_ema_t1_t180'],1,0)

forex_df['lag_t1_trend_360'] = np.where(forex_df['rolling_ema_t1_t7'] >= forex_df['rolling_ema_t1_t360'],1,0)
#number of crossovers last n days

forex_df['lag_t1_no_crossover_7'] = forex_df['lag_t1_trend_7'].rolling(window=7,min_periods=1).sum()

forex_df['lag_t1_no_crossover_14'] = forex_df['lag_t1_trend_14'].rolling(window=14,min_periods=1).sum()

forex_df['lag_t1_no_crossover_28'] = forex_df['lag_t1_trend_28'].rolling(window=28,min_periods=1).sum()

forex_df['lag_t1_no_crossover_90'] = forex_df['lag_t1_trend_90'].rolling(window=90,min_periods=1).sum()

forex_df['lag_t1_no_crossover_180'] = forex_df['lag_t1_trend_180'].rolling(window=180,min_periods=1).sum()

forex_df['lag_t1_no_crossover_360'] = forex_df['lag_t1_trend_360'].rolling(window=360,min_periods=1).sum()
#decay
#slope or 1st derivative

forex_df['lag_t1_slope_7'] = forex_df['lag_t1'].rolling(7).apply(lambda x: np.polyfit(range(7), x, 1)[0]).values

forex_df['lag_t1_slope_14'] = forex_df['lag_t1'].rolling(14).apply(lambda x: np.polyfit(range(14), x, 1)[0]).values

forex_df['lag_t1_slope_28'] = forex_df['lag_t1'].rolling(28).apply(lambda x: np.polyfit(range(28), x, 1)[0]).values

forex_df['lag_t1_slope_90'] = forex_df['lag_t1'].rolling(90).apply(lambda x: np.polyfit(range(90), x, 1)[0]).values

forex_df['lag_t1_slope_180'] = forex_df['lag_t1'].rolling(180).apply(lambda x: np.polyfit(range(180), x, 1)[0]).values

forex_df['lag_t1_slope_360'] = forex_df['lag_t1'].rolling(360).apply(lambda x: np.polyfit(range(360), x, 1)[0]).values
#2nd derivative, slope of the 1st derivative, again for detecting trend changes

forex_df['lag_t1_deriv2_7'] = forex_df['lag_t1_slope_7'].rolling(7).apply(lambda x: np.polyfit(range(7), x, 1)[0]).values

forex_df['lag_t1_deriv2_14'] = forex_df['lag_t1_slope_7'].rolling(14).apply(lambda x: np.polyfit(range(14), x, 1)[0]).values

forex_df['lag_t1_deriv2_28'] = forex_df['lag_t1_slope_7'].rolling(28).apply(lambda x: np.polyfit(range(28), x, 1)[0]).values

forex_df['lag_t1_deriv2_90'] = forex_df['lag_t1_slope_7'].rolling(90).apply(lambda x: np.polyfit(range(90), x, 1)[0]).values

forex_df['lag_t1_deriv2_180'] = forex_df['lag_t1_slope_7'].rolling(180).apply(lambda x: np.polyfit(range(180), x, 1)[0]).values

forex_df['lag_t1_deriv2_360'] = forex_df['lag_t1_slope_7'].rolling(360).apply(lambda x: np.polyfit(range(360), x, 1)[0]).values

forex_df.shape
#We have a heap of features:

list(forex_df.columns)


#Create a list of the features to drop, as previously mentioned we can't use the feature from today else it would cause target leakage - the model knows something that it can't know in advance.

useless_cols = ['Unnamed: 0', 

                "date", 

                'AUSTRALIA - AUSTRALIAN DOLLAR/US$',

                'Time Serie', 

                'EURO AREA - EURO/US$',

                 'NEW ZEALAND - NEW ZELAND DOLLAR/US$',

                 'UNITED KINGDOM - UNITED KINGDOM POUND/US$',

                 'BRAZIL - REAL/US$',

                 'CANADA - CANADIAN DOLLAR/US$',

                 'CHINA - YUAN/US$',

                 'HONG KONG - HONG KONG DOLLAR/US$',

                 'INDIA - INDIAN RUPEE/US$',

                 'KOREA - WON/US$',

                 'MEXICO - MEXICAN PESO/US$',

                 'SOUTH AFRICA - RAND/US$',

                 'SINGAPORE - SINGAPORE DOLLAR/US$',

                 'DENMARK - DANISH KRONE/US$',

                 'JAPAN - YEN/US$',

                 'MALAYSIA - RINGGIT/US$',

                 'NORWAY - NORWEGIAN KRONE/US$',

                 'SWEDEN - KRONA/US$',

                 'SRI LANKA - SRI LANKAN RUPEE/US$',

                 'SWITZERLAND - FRANC/US$',

                 'TAIWAN - NEW TAIWAN DOLLAR/US$',

                 'THAILAND - BAHT/US$']



#define train columns to use in model

train_cols = forex_df.columns[~forex_df.columns.isin(useless_cols)]





 



#Let's simply use historical data up until Oct 2019

x_train = forex_df[forex_df['date'] <= '2019-10-31'].copy()

#The variable we want to predict is AUD to USD rate.

y_train = x_train['AUSTRALIA - AUSTRALIAN DOLLAR/US$']



#The LGBM model needs a train and validation dataset to be fed into it, let's use Nov 2019

x_val = forex_df[(forex_df['date'] > '2019-10-31') & (forex_df['date'] <= '2019-11-30')].copy()

y_val = x_val['AUSTRALIA - AUSTRALIAN DOLLAR/US$']



#We shall test the model on data it hasn't seen before or been used in the training process

test = forex_df[(forex_df['date'] > '2019-12-01')].copy()



#Setup the data in the necessary format the LGB requires

train_set = lgb.Dataset(x_train[train_cols], y_train)

val_set = lgb.Dataset(x_val[train_cols], y_val)
#Set the model parameters

params = {

        "objective" : "regression", # regression is the type of business case we are running

        "metric" :"rmse", #root mean square error is a standard metric to use

        "learning_rate" : 0.05, #the pace at which the model is allowed to reach it's objective of minimising the rsme.

        'num_iterations' : 2000,

        'num_leaves': 50, # minimum number of leaves in each boosting round

        "early_stopping": 50, #if the model does not improve after this many consecutive rounds, call a halt to training

        "max_bin": 200,

        "seed":888



}
#Run the model

m1_lgb = lgb.train(params, train_set, num_boost_round = 2500, valid_sets = [train_set, val_set], verbose_eval = 50)
#plot feature importance

feature_imp = pd.DataFrame({'Value':m1_lgb.feature_importance(),'Feature':train_cols})

plt.figure(figsize=(20, 10))

sns.set(font_scale = 1)

sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", 

                                                    ascending=False)[0:40])

plt.title('LightGBM Features (avg over folds)')

plt.tight_layout()

plt.savefig('lgbm_importances-01.png')

plt.show()
#generate predictions on test data

y_pred = m1_lgb.predict(test[train_cols])

test['AUSTRALIA - AUSTRALIAN DOLLAR/US$_pred'] = y_pred
#view the test data in chart form

df_long=pd.melt(test, id_vars=['date'], value_vars=['AUSTRALIA - AUSTRALIAN DOLLAR/US$', 'AUSTRALIA - AUSTRALIAN DOLLAR/US$_pred'])



# plotly 

fig = px.line(df_long, x='date', y='value', color='variable')



# Show plot 

fig.show()
#RMSE metric

rms_m1 = sqrt(mean_squared_error(test['AUSTRALIA - AUSTRALIAN DOLLAR/US$'], test['AUSTRALIA - AUSTRALIAN DOLLAR/US$_pred']))
rms_m1
from sklearn import preprocessing, metrics

from sklearn.model_selection import KFold, TimeSeriesSplit
n_fold = 100

folds = TimeSeriesSplit(n_splits=n_fold)

splits = folds.split(x_train, y_train)



y_preds = np.zeros(test.shape[0])

y_oof = np.zeros(x_train.shape[0])

feature_importances = pd.DataFrame()

feature_importances['feature'] = train_cols

mean_score = []
#Set the model parameters

params = {

        "objective" : "regression", # regression is the type of business case we are running

        "metric" :"rmse", #root mean square error is a standard metric to use

        "learning_rate" : 0.05, #the pace at which the model is allowed to reach it's objective of minimising the rsme.

        'num_iterations' : 2000,

        'num_leaves': 50, # minimum number of leaves in each boosting round

        "early_stopping": 50, #if the model does not improve after this many consecutive rounds, call a halt to training

        "max_bin": 200,

        "seed":888



}
#Include the additional month reserved from validation under the previous method

x_train_tss = forex_df[forex_df['date'] <= '2019-11-30'].copy() #changed month from Oct to Nov

y_train_tss = x_train_tss['AUSTRALIA - AUSTRALIAN DOLLAR/US$'] 
for fold_n, (train_index, valid_index) in enumerate(splits):

    print('Fold:',fold_n+1)

    #generate the train and validation datasets for each fold

    X_train1, X_valid1 = x_train_tss[train_cols].iloc[train_index], x_train_tss[train_cols].iloc[valid_index]

    y_train1, y_valid1 = y_train_tss.iloc[train_index], y_train_tss.iloc[valid_index]

    #convert the data into lgb format 

    dtrain = lgb.Dataset(X_train1, label=y_train1)

    dvalid = lgb.Dataset(X_valid1, label=y_valid1)

    #train lgb model, using same parameters as previous model for more direct comparison

    clf = lgb.train(params, dtrain, valid_sets = [dtrain, dvalid], verbose_eval=100)

    #feature importance

    feature_importances[f'fold_{fold_n + 1}'] = clf.feature_importance()

    #make predictions on validation set

    y_pred_valid = clf.predict(X_valid1,num_iteration=clf.best_iteration)

    y_oof[valid_index] = y_pred_valid

    #validation score

    val_score = np.sqrt(metrics.mean_squared_error(y_pred_valid, y_valid1))

    print(f'val rmse score is {val_score}')

    mean_score.append(val_score)

    y_preds += clf.predict(test[train_cols], num_iteration=clf.best_iteration)/n_fold

    del X_train1, X_valid1, y_train1, y_valid1

    gc.collect()

print('mean rmse score over folds is',np.mean(mean_score))
feature_importances
#generate predictions on test data

y_pred_ts_split = clf.predict(test[train_cols])

test['AUSTRALIA - AUSTRALIAN DOLLAR/US$_pred_ts_split'] = y_pred_ts_split


df_long=pd.melt(test, id_vars=['date'], value_vars=['AUSTRALIA - AUSTRALIAN DOLLAR/US$', 'AUSTRALIA - AUSTRALIAN DOLLAR/US$_pred','AUSTRALIA - AUSTRALIAN DOLLAR/US$_pred_ts_split'])



# plotly 

fig = px.line(df_long, x='date', y='value', color='variable')



# Show plot 

fig.show()
#RSME metric

rms_ts_split = sqrt(mean_squared_error(test['AUSTRALIA - AUSTRALIAN DOLLAR/US$'], test['AUSTRALIA - AUSTRALIAN DOLLAR/US$_pred_ts_split']))
rms_ts_split