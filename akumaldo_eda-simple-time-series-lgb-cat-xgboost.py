# Python ≥3.5 is required

import sys

assert sys.version_info >= (3, 5)



# Scikit-Learn ≥0.20 is required

import sklearn

assert sklearn.__version__ >= "0.20"



# Common imports

import numpy as np

import os

import gc



import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import pandas as pd



%matplotlib inline 

#plotting directly without requering the plot()



import warnings

warnings.filterwarnings(action="ignore") #ignoring most of warnings, cleaning up the notebook for better visualization



pd.set_option('display.max_columns', 500) #fixing the number of rows and columns to be displayed

pd.set_option('display.max_rows', 500)



print(os.listdir("../input")) #showing all the files in the ../input directory



# Any results you write to the current directory are saved as output. Kaggle message :D

# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage

def reduce_mem_usage(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.        

    """

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))



    for col in df.columns:

        col_type = df[col].dtype



        if col_type != object and col_type != '<M8[ns]':

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

            elif col_type != '<M8[ns]':

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))



    return df

sales_train = pd.read_csv("../input/sales_train.csv", parse_dates=True, index_col=0, infer_datetime_format=True, dayfirst=True)

test = pd.read_csv("../input/test.csv")

shop_df = pd.read_csv("../input/shops.csv")

items = pd.read_csv('../input/items.csv')

item_cat = pd.read_csv("../input/item_categories.csv") #reading the file
print(sales_train.shape, test.shape)
print("The columns in the training set are: %s" %list(sales_train.columns))

print("The columns in the testing set are: %s" %list(test.columns))
sales_train.head() #looking at the first entries of our training set
sales_train.isnull().sum() #checking whether we have null values or not.
print('Total number of shopping(by ID): %d' %sales_train['shop_id'].max()) #number of different shop's ID

print('Number of months: %d' %sales_train['date_block_num'].max()) #number of months 

print('Total number of items(by ID): %d' %sales_train['item_price'].max()) #numer of different item's ID
sales_train = sales_train[sales_train['item_cnt_day'] > 0] #keeping only items with price bigger than 0
item_cat.head()
print('Number of categories: %s' %item_cat['item_category_name'].nunique()) #checking for unique names

print('number of categories id: %s' %str(item_cat['item_category_id'].max())) #checking for number of ids

print('Shape of item categories dataset: %s' %str(item_cat.shape))
items.head()
print('Number of items: %s' %items['item_id'].nunique()) #printing the number of unique items
shop_df.head()
print('Total number of shops: %s' %shop_df['shop_id'].nunique()) 

print('Shape: %s' %str(shop_df.shape))
#### Joining first item categories, using item id

items = items.join(item_cat, on="item_category_id", rsuffix='_')

items.head()
#### now joining sales

sales_train = sales_train.join(items, on="item_id", rsuffix="_")

sales_train.head()
#### Now finally, join with shopping list, then clear the memory up

sales_train = sales_train.join(shop_df, on="shop_id", rsuffix="_")

train = sales_train.drop(["shop_id_","item_id_","item_category_id_"], axis=1) #dropping all duplicate IDs(using the suffix "_")

train.head()
# drop shops&items not in test data

shop_id = test.shop_id.unique()

item_id = test.item_id.unique()

train = train[train.shop_id.isin(shop_id)]

train = train[train.item_id.isin(item_id)]

####Cleaning the memory up####

gc.enable

del items, item_cat, shop_df, sales_train,shop_id,item_id

gc.collect()
train_by_month = train.reset_index()[['date', 'date_block_num', 'shop_id', 'item_category_id', 'item_id', 'item_price', 'item_cnt_day']]
train_by_month = train_by_month.sort_values('date').groupby(['date_block_num', 'shop_id', 'item_category_id','item_id'], as_index=False)

train_by_month = train_by_month.agg({'item_price':['sum', 'mean'], 'item_cnt_day':['sum', 'mean','count']})

train_by_month.columns = ['date_block_num', 'shop_id', 'item_category_id', 'item_id','item_price', 'mean_item_price', 'item_cnt', 'mean_item_cnt', 'transactions']

train_by_month.head()
shop_ids = train_by_month['shop_id'].unique()

item_ids = train_by_month['item_id'].unique()

empty_df = []

for i in range(34):

    for shop in shop_ids:

        for item in item_ids:

            empty_df.append([i, shop, item])

    

empty_df = pd.DataFrame(empty_df, columns=['date_block_num','shop_id','item_id'])

train_by_month = pd.merge(empty_df, train_by_month, on=['date_block_num','shop_id','item_id'], how='left')

train_by_month['year'] = train_by_month['date_block_num'].apply(lambda x: ((x//12) + 2013)) #creating year and month using the data_block_num, a better solution would be to use datetime, but as we have created different entries above, I couldn't came up with a different solution

train_by_month['month'] = train_by_month['date_block_num'].apply(lambda x: (x % 12))

train_by_month.fillna(0, inplace=True) #filling with zero, information we don't have.
train_by_month = reduce_mem_usage(train_by_month) #reducing the memory usage, changing the features types, using the function implemented at the beggining
plt.style.use('fivethirtyeight')

plt.figure(figsize = (16, 12))



cnt_item_by_month = train_by_month.groupby(['month',"shop_id","item_id"],as_index=False)['item_cnt'].sum()

cnt_ctg_by_month = train_by_month.groupby(['month','item_category_id'], as_index=False)['item_cnt'].sum()



rand_item_id = [30,30,22167,22167] ## 'randomly' picking examples

rand_shop_id = [3,5,28,42]

num_plots = len(rand_item_id)

i=0

# iterate through the sources

for item, shop in zip(rand_item_id,rand_shop_id):

    # create a new subplot for each source

    plt.subplot(num_plots, 1, i + 1)

    temp = cnt_item_by_month[["shop_id","item_id","month","item_cnt"]]

    temp = cnt_item_by_month.loc[(cnt_item_by_month['item_id'] == item) & (cnt_item_by_month['shop_id'] == shop)]

    plt.plot(temp["month"],temp["item_cnt"]);



    plt.title('Sum of items per year_month -- item id: %s -- shop id: %s' %(item,shop)); plt.xlabel('month'); plt.ylabel('Count')

    i+=1

plt.tight_layout(h_pad = 1.5)



#cleaning out the memory

gc.enable

del rand_item_id,rand_shop_id,temp, num_plots, shop_ids,item_ids,empty_df

gc.collect
sns.set_style('ticks')

plt.figure(figsize = (16, 5))

plt.subplot(2,1,1)

sns.lineplot(x='month',y='item_cnt', data=cnt_item_by_month);plt.xticks(fontsize=10); plt.title('Item sales per month')

plt.subplot(2,1,2)

sns.barplot(x='item_category_id',y='item_cnt', data=cnt_ctg_by_month,errwidth=0);plt.xticks(fontsize=10); plt.title('Item count per each category')

plt.tight_layout(h_pad = 1.5)
plt.figure(figsize = (16, 5))



sns.barplot(x='shop_id',y='item_cnt', data=cnt_item_by_month,errwidth=0);plt.xticks(fontsize=10); plt.title('Item count per shopping id')
ax, fig = plt.subplots(1,1, figsize=(15,5))

plt.subplot(2,1,1)

sns.boxplot(x='date_block_num', y='item_price', data=train_by_month)

plt.subplot(2,1,2)

sns.boxplot(x='date_block_num', y='item_cnt', data=train_by_month)
train_by_month[train_by_month['item_price'] > 500000].count() #checking for the item_price higher than 500000
train_by_month[train_by_month['item_cnt'] > 20].count() #checking for item count above 20
train_by_month = train_by_month.query('item_price < 500000 and item_cnt >= 0 and item_cnt <= 20')
train_by_month['item_cnt_month'] = train_by_month.sort_values('date_block_num').groupby(['shop_id', 'item_id'])['item_cnt'].shift(-1)

train_by_month.head()
train_by_month.tail()
print("The shape of the training data before feature engineering: {}".format(train_by_month.shape))
# Min value

r_min = lambda x: x.rolling(window=3, min_periods=1).min()

# Max value

r_max = lambda x: x.rolling(window=3, min_periods=1).max()

# Mean value

r_mean = lambda x: x.rolling(window=3, min_periods=1).mean()

# Standard deviation

r_std = lambda x: x.rolling(window=3, min_periods=1).std()



function_list = [r_min, r_max, r_mean, r_std] #list with the each function above listed that is gonna be applied taking the last 3 months

function_name = ['min', 'max', 'mean', 'std'] #names of the functions



for i in range(len(function_list)):

    train_by_month[('item_cnt_%s' % function_name[i])] = train_by_month.sort_values('date_block_num').groupby(['shop_id', 'item_category_id', 'item_id'])['item_cnt'].apply(function_list[i])



# Fill the empty std features with 0

train_by_month['item_cnt_std'].fillna(0, inplace=True)
lag_list = [1, 2, 3] #creating a lag list with each month, 1, 2 and 3 months later.



for lag in lag_list: #going through the list of months

    ft_name = ('item_cnt_%s_month_before' % lag) # lag number of months before, getting the previous item count per month

    train_by_month[ft_name] = train_by_month.sort_values('date_block_num').groupby(['shop_id', 'item_category_id', 'item_id'])['item_cnt'].shift(lag)

    # Fill the empty shifted features with 0

    train_by_month[ft_name].fillna(0, inplace=True)
train_by_month['item_trend'] = train_by_month['item_cnt'] #firstly, the item trend is equal to the item count.



for lag in lag_list: #searching for each lag feature, then subtracting it from the trend value.

    ft_name = ('item_cnt_%s_month_before' % lag)

    train_by_month['item_trend'] -= train_by_month[ft_name]



train_by_month['item_trend'] /= len(lag_list) + 1 #finally, dividing it by 3 months + 1

train_by_month.head()
train_final = train_by_month.query('date_block_num >= 3 and date_block_num < 28').copy()

train_validation_final = train_by_month.query('date_block_num >= 28 and date_block_num < 33').copy()



train_final.dropna(inplace=True) #dropping any NaN values

train_validation_final.dropna(inplace=True) #dropping any NaN values

 

#dropping item category ID as we don't have it in our testing set.

train_final.drop(['item_category_id'], axis=1, inplace=True) 

train_validation_final.drop(['item_category_id'], axis=1, inplace=True)



print("Training set: {}".format(train_final.shape)+"\nValidation set: {}"

      .format(train_validation_final.shape))
#grouping by shopping id, item count per month by shopping ID

gp_shop_mean = train_final.groupby(['shop_id']).agg({'item_cnt_month': ['mean']})

gp_shop_mean.columns = ['shop_mean']

gp_shop_mean.reset_index(inplace=True)

#grouping by item ID, item count per month by item ID

gp_item_mean = train_final.groupby(['item_id']).agg({'item_cnt_month': ['mean']})

gp_item_mean.columns = ['item_mean']

gp_item_mean.reset_index(inplace=True)

#grouping by shopping id and item ID, item count per month by shopping ID and item ID

gp_shop_item_mean = train_final.groupby(['shop_id', 'item_id']).agg({'item_cnt_month': ['mean']})

gp_shop_item_mean.columns = ['shop_item_mean']

gp_shop_item_mean.reset_index(inplace=True)

#grouping by year and getting the item cnt per month mean

gp_year_mean = train_final.groupby(['year']).agg({'item_cnt_month': ['mean']})

gp_year_mean.columns = ['year_mean']

gp_year_mean.reset_index(inplace=True)

#grouping by month and getting the item cnt per month mean

gp_month_mean = train_final.groupby(['month']).agg({'item_cnt_month': ['mean']})

gp_month_mean.columns = ['month_mean']

gp_month_mean.reset_index(inplace=True)



# Merging the features created into the train final dataset

train_final = pd.merge(train_final, gp_shop_mean, on=['shop_id'], how='left')

train_final = pd.merge(train_final, gp_item_mean, on=['item_id'], how='left')

train_final = pd.merge(train_final, gp_shop_item_mean, on=['shop_id', 'item_id'], how='left')

train_final = pd.merge(train_final, gp_year_mean, on=['year'], how='left')

train_final = pd.merge(train_final, gp_month_mean, on=['month'], how='left')

# Merging the features created into the validation dataset

train_validation_final = pd.merge(train_validation_final, gp_shop_mean, on=['shop_id'], how='left')

train_validation_final = pd.merge(train_validation_final, gp_item_mean, on=['item_id'], how='left')

train_validation_final = pd.merge(train_validation_final, gp_shop_item_mean, on=['shop_id', 'item_id'], how='left')

train_validation_final = pd.merge(train_validation_final, gp_year_mean, on=['year'], how='left')

train_validation_final = pd.merge(train_validation_final, gp_month_mean, on=['month'], how='left')



#finally, adding those features created to our testing set as well

additional_features = pd.concat([train_final, train_validation_final]).drop_duplicates(subset=['shop_id', 'item_id'], keep='last')

test_final = pd.merge(test, additional_features, on=['shop_id', 'item_id'], how='left', suffixes=['', '_'])

test_final['year'] = 2015 #setting the month and year manually, we are predicting the next month, setting the current month to 9 then

test_final['month'] = 9

test_final = test_final[train_final.columns]#selecting only the columns present in the training set, keeping it aligned



print("Training set: {}".format(train_final.shape)+"\nValidation set: {}"

      .format(train_validation_final.shape) + "\nTesting set: {}".format(test_final.shape))
#this pipeline is gonna be use for numerical atributes and standard scaler

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler,RobustScaler, MinMaxScaler



num_pipeline = Pipeline([

        ('imputer', SimpleImputer(strategy="median")),

        #('robust_scaler', RobustScaler()),

        ('minmax_scaler', MinMaxScaler(feature_range=(0, 1)))

    ])
train_labels = train_final['item_cnt_month'].astype(int)

train_final.drop(['date_block_num','item_cnt_month'], axis=1,inplace=True)

train_val_labels = train_validation_final['item_cnt_month'].astype(int)

train_validation_final.drop(['date_block_num','item_cnt_month'], axis=1,inplace=True)



#dropping the target in our testing set

test_final.drop(['date_block_num','item_cnt_month'], axis=1,inplace=True)



print("Training set: {}".format(train_final.shape)+"\nValidation set: {}"

      .format(train_validation_final.shape) + "\nTesting set: {}".format(test_final.shape))
from sklearn.metrics import mean_squared_error

import time #implementing in this function the time spent on training the model

from catboost import CatBoostRegressor

from catboost import Pool

import lightgbm as lgb

import xgboost as xgb

import gc



cols = ['item_cnt','item_cnt_mean', 'item_cnt_std', 'item_cnt_1_month_before', 

                'item_cnt_2_month_before', 'item_cnt_3_month_before', 'shop_mean', 

                'shop_item_mean', 'item_trend', 'mean_item_cnt','transactions','year','month']



#Selecting only relevant features(I ran the model before and selected the top features only), 

#ignoring item id, shopping id and using the pipeline to scale the data



def train_model(X, X_val, y, y_val, params=None, model_type='lgb', plot_feature_importance=False):

  

    evals_result={}

    

    X_train = num_pipeline.fit_transform(X.loc[:,cols])

    x_val = num_pipeline.fit_transform(X_val.loc[:,cols])

    

    if model_type == 'lgb':

        start = time.time()

        

        model = lgb.LGBMRegressor(**params, n_estimators = 10000, nthread = 4, n_jobs = -1)

        

        model.fit(X_train, y, eval_set=[(X_train, y), (x_val, y_val)], eval_metric='rmse', early_stopping_rounds=200,

                    verbose=50)

            

        y_pred_valid = model.predict(x_val, num_iteration=model.best_iteration_)

        

        end = time.time()

        

        #y_pred = model.predict(X_test, num_iteration=model.best_iteration_)

        

        print('RMSE validation data: {}'.format(np.sqrt(mean_squared_error(y_val,y_pred_valid))))

        

        

        if plot_feature_importance:

            # feature importance

            fig, ax = plt.subplots(figsize=(12,10))

            lgb.plot_importance(model, max_num_features=50, height=0.8,color='c', ax=ax)

            ax.grid(False)

            plt.title("LightGBM - Feature Importance", fontsize=15)

            

        print('Total time spent: {}'.format(end-start))

        return model

            

    if model_type == 'xgb':

        start = time.time()

        

        model = xgb.XGBRegressor(**params, nthread = 4, n_jobs = -1)



        model.fit(X_train, y, eval_metric="rmse", 

                      eval_set=[(X_train, y), (x_val, y_val)],verbose=20,

                      early_stopping_rounds=50)

        

        y_pred_valid = model.predict(x_val, ntree_limit=model.best_ntree_limit)

        

        end = time.time()



        print('RMSE validation data: {}'.format(np.sqrt(mean_squared_error(y_val,y_pred_valid))))

        

        print('Total time spent: {}'.format(end-start))

        return model

            

    if model_type == 'cat':

        start = time.time()

        model = CatBoostRegressor(eval_metric='RMSE', **params)

        model.fit(X_train, y, eval_set=(x_val, y_val), 

                  cat_features=[], use_best_model=True)



        y_pred_valid = model.predict(x_val)

        

        print('RMSE validation data: {}'.format(np.sqrt(mean_squared_error(y_val,y_pred_valid))))

        

        end = time.time()

        

        if plot_feature_importance:

            feature_score = pd.DataFrame(list(zip(X.loc[:,cols].dtypes.index, model.get_feature_importance(Pool(X.loc[:,cols], label=y, cat_features=[])))), columns=['Feature','Score'])

            feature_score = feature_score.sort_values(by='Score', kind='quicksort', na_position='last')

            feature_score.plot('Feature', 'Score', kind='barh', color='c', figsize=(16,10))

            plt.title("Catboost Feature Importance plot", fontsize = 14)

            plt.xlabel('')



        print('Total time spent: {}'.format(end-start))

        return model

        

    # Clean up memory

    gc.enable()

    del model, y_pred_valid, X_test,X_train,X_valid, y_pred, y_train, start, end,evals_result, x_val

    gc.collect()

params_cat = {

    'iterations': 1000,

    'max_ctr_complexity': 4,

    'random_seed': 42,

    'od_type': 'Iter',

    'od_wait': 100,

    'verbose': 50,

    'depth': 4

}



cat_model = train_model(train_final,train_validation_final,train_labels,train_val_labels,params_cat,model_type='cat',plot_feature_importance=True)
params_lgb = {

        "objective" : "regression",

        "metric" : "rmse",

        "num_leaves" : 30,

        "min_child_weight" : 50,

        "learning_rate" : 0.009,

        "bagging_fraction" : 0.7,

        "feature_fraction" : 0.7,

        "bagging_frequency" : 5,

        "bagging_seed" : 42,

}



lgb_model = train_model(train_final,train_validation_final,train_labels,train_val_labels,params_lgb)
params_xgb = {

    "max_depth": 8,

    "n_estimators": 5000,

    "learning_rate" : 0.05,

    "min_child_weight": 1000,  

    "colsample_bytree": 0.7, 

    "subsample": 0.7, 

    "eta": 0.3, 

    "seed": 42

}



xgb_model = train_model(train_final,train_validation_final,train_labels,train_val_labels,params_xgb,model_type="xgb")
#preparing the test dataset and passing it to each model...

test = num_pipeline.fit_transform(test_final.loc[:,cols])



prediction_cat = cat_model.predict(test)

prediction_lgb = lgb_model.predict(test)

prediction_xgb = xgb_model.predict(test)



sub = pd.read_csv('../input/sample_submission.csv')

final_prediction = (prediction_cat+prediction_lgb+prediction_xgb)/3

sub['item_cnt_month'] = final_prediction.clip(0., 20.)

sub.to_csv('mixed_sub.csv', index=False)

sub.head()
cols = ['item_cnt','item_cnt_mean', 'item_cnt_std', 'item_cnt_1_month_before',

        'item_cnt_2_month_before', 'item_cnt_3_month_before', 'shop_mean',

        'shop_item_mean', 'item_trend', 'mean_item_cnt','transactions','year','month']

X_train = num_pipeline.fit_transform(train_final.loc[:,cols])

X_val = num_pipeline.fit_transform(train_validation_final.loc[:,cols])

X_train = np.expand_dims(X_train, axis=2)

X_val = np.expand_dims(X_val, axis=2)



#### importing relevant models ####

from keras.models import Sequential

from keras.callbacks import EarlyStopping

from keras.layers import LSTM, Dense, Activation, ThresholdedReLU, MaxPooling2D, Embedding, Dropout

from keras.optimizers import Adam, SGD, RMSprop



# Defining the model layers

model_lstm = Sequential()

model_lstm.add(LSTM(16, input_shape=(X_train.shape[1],X_train.shape[2]),return_sequences=True))

model_lstm.add(Dropout(0.5))

model_lstm.add(LSTM(32))

model_lstm.add(Dropout(0.5))

model_lstm.add(Dense(1))

model_lstm.compile(optimizer="adam", loss='mse', metrics=["mse"])

print(model_lstm.summary())





params_lstm = {"batch_size":64,

              "verbose":2,

              "epochs":10}



callbacks_list=[EarlyStopping(monitor="val_loss",min_delta=.001, patience=3,mode='auto')]

hist = model_lstm.fit(X_train, train_labels,

                      validation_data=(X_val, train_val_labels),

                      callbacks=callbacks_list,**params_lstm)
fig, ax = plt.subplots(2,1,figsize=(12,10))

ax[0].plot(hist.history['loss'], color='b', label="Training loss")

ax[0].plot(hist.history['val_loss'], color='r', label="validation loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(hist.history['mean_squared_error'], color='b', label="Training mean squared error")

ax[1].plot(hist.history['val_mean_squared_error'], color='r',label="Validation mean squared error")

legend = ax[1].legend(loc='best', shadow=True)
# predict results

X_test = np.expand_dims(test, axis=2)

results = model_lstm.predict(X_test)

sub['item_cnt_month'] = results.clip(0., 20.)

sub.to_csv('lstm.csv', index=False)

sub.head()