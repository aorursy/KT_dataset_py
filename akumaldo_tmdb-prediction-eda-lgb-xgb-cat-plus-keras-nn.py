# Python ≥3.5 is required

import sys

assert sys.version_info >= (3, 5)



# Scikit-Learn ≥0.20 is required

import sklearn

assert sklearn.__version__ >= "0.20"



# Common imports

import numpy as np

import os



from collections import Counter

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import pandas as pd

import ast



%matplotlib inline 

#plotting directly without requering the plot()



import warnings

warnings.filterwarnings(action="ignore")



pd.set_option('display.max_columns', 500) #fixing the number of rows and columns to be displayed

pd.set_option('display.max_rows', 500)



print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage

def reduce_mem_usage(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.        

    """

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))



    for col in df.columns:

        col_type = df[col].dtype



        if col_type != object:

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

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))



    return df
dict_columns = ['belongs_to_collection', 'genres', 'production_companies',

                'production_countries', 'spoken_languages', 'Keywords', 'cast', 'crew']



def text_to_dict(df):

    for column in dict_columns:

        df[column] = df[column].apply(lambda x: {} if pd.isna(x) else ast.literal_eval(x) )

    return df

        

train = text_to_dict(train)

test = text_to_dict(test)
print(train.shape, test.shape) #looking at the shape of our dataframes
train.head() #taking a look at the first entries
##Merging the train and test dataset in order to have more data to train our model.



train['source']='train' #creating a label for the training and testing set

test['source']='test'



data = pd.concat([train, test],ignore_index=True)

print (train.shape, test.shape, data.shape) #printing the shape
print("Belongs to a collection or not:\n%s" % (data['belongs_to_collection'].apply(lambda x: len(x) if x != {} else 0).value_counts()))
## Function that receives the dataset, reads the dictionary and create a new boolean column for each top 20 common value

def dict_one_hot_code(dataset,name_column, name_new_column):

    list_of_values = list(dataset[name_column].apply(lambda x: [i['name'] for i in x] if x != {} else []).values) #create a list of each occurence of a given name

    Counter([i for j in list_of_values for i in j]).most_common(20)

    top_list_values = [m[0] for m in Counter([i for j in list_of_values for i in j]).most_common(20)]

    for g in top_list_values: 

        dataset[name_new_column + "_" + g] = dataset[name_column].apply(lambda x: 1 if g in str(x) else 0)

        

def counting_number_of_values_dict(dataset, name_column):

    return dataset[name_column].apply(lambda x: len(x) if x!= {} else 0)
data['collection_name'] = data['belongs_to_collection'].apply(lambda x: x[0]['name'] if x != {} else "None")

data['is_part_of_collection'] = data['belongs_to_collection'].apply(lambda x: len(x) if x != {} else 0) 

#feature engineering a new column that has the len of the belongs_to_collection columns, by doing so, you attribute a weight to each particular movie, or zero if it doesn't belong to any



data.drop('belongs_to_collection', axis = 1, inplace=True)
data.head() #checking the changes 
#count the number of genres

data['genres'].apply(lambda x: x[0]['name'] if x != {} else 0).value_counts() #this counts for us the number of dicts containing the genre in our dataset and count the number of occurences
data['genre'] = data['genres'].apply(lambda x: x[0]['name'] if x != {} else "None") # exactly the same code used for the collection belongs.

data.drop('genres', axis=1, inplace=True)
data['cast'][0]
data['cast'][0][0]
list_of_cast_names = list(data['cast'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values) #create a list of each occurence of a given name

Counter([i for j in list_of_cast_names for i in j]).most_common(20) #count every instance of each actor and sum them all to return back the 20 most common instances
#and let's also take a look at the total number of cast member for each movie

data['cast'].apply(lambda x: len(x) if x != {} else 0).value_counts()
data['number_cast'] = counting_number_of_values_dict(data,'cast')
data.head()
list_of_cast_characters = list(data['cast'].apply(lambda x: [i['character'] for i in x] if x != {} else []).values)

Counter([i for j in list_of_cast_characters for i in j]).most_common(15)
top_cast_names = [m[0] for m in Counter([i for j in list_of_cast_names for i in j]).most_common(15)]

for g in top_cast_names: 

    data['cast_name_' + g] = data['cast'].apply(lambda x: 1 if g in str(x) else 0) #creating a boolean for the top 15 most common actors...
data.head()
data.shape
top_cast_characters = [m[0] for m in Counter([i for j in list_of_cast_characters for i in j]).most_common(15)]

for g in top_cast_characters:

    data['cast_character_' + g] = data['cast'].apply(lambda x: 1 if g in str(x) else 0)
data.drop('cast', axis=1, inplace=True)
counting_number_of_values_dict(data,'crew').value_counts()
data['crew_total'] = counting_number_of_values_dict(data, 'crew')
dict_one_hot_code(data,name_column='crew',name_new_column='crew_name')
data.drop('crew', axis=1, inplace=True)
data.shape
data.head()
data['Keywords'][0]
counting_number_of_values_dict(data,'Keywords').value_counts()
data['keywords_num'] = counting_number_of_values_dict(data,'Keywords')
list_of_keywords = list(data['Keywords'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)
data['all_Keywords'] = data['Keywords'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')
top_keywords = [m[0] for m in Counter([i for j in list_of_keywords for i in j]).most_common(30)]

for g in top_keywords:

    data['keyword_' + g] = data['all_Keywords'].apply(lambda x: 1 if g in x else 0)
data.drop(['Keywords','all_Keywords'], axis=1, inplace=True)
counting_number_of_values_dict(data,'spoken_languages').value_counts()
languages = list(data['spoken_languages'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

Counter([i for j in languages for i in j]).most_common(15)
data['num_languages'] = counting_number_of_values_dict(data,'spoken_languages')

data['all_languages'] = data['spoken_languages'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')

top_languages = [m[0] for m in Counter([i for j in languages for i in j]).most_common(30)]

for g in top_languages:

    data['language_' + g] = data['all_languages'].apply(lambda x: 1 if g in x else 0)
data.drop(['spoken_languages', 'all_languages'], axis=1, inplace=True)
data.shape #number of features has increased significantly
counting_number_of_values_dict(data,'production_countries').value_counts()
data['num_countries'] = counting_number_of_values_dict(data, 'production_countries')

dict_one_hot_code(data,name_column= 'production_countries', name_new_column='production_country')

data.head()
data.drop('production_countries', axis=1, inplace=True) #and finally, dropping the dictionary...
counting_number_of_values_dict(data,'production_companies').value_counts()
data['prod_companies_total_number'] = counting_number_of_values_dict(data,'production_companies')
dict_one_hot_code(data,'production_companies','all_production_companies')
data.drop('production_companies', axis=1, inplace=True)
data.shape
data.head()
data['homepage'].isnull().value_counts() 
data['has_homepage'] = 0 #setting all the values to false as default

data.loc[data['homepage'].isnull() == False, 'has_homepage'] = 1

data.drop('homepage', axis=1, inplace=True)
# Set the style of plots

plt.style.use('fivethirtyeight')



fig, ax = plt.subplots(2,2, figsize = (16, 6))



plt.subplot(1, 2, 1)

sns.countplot(y=data['genre'])

plt.title('Distribution of genres')

plt.subplot(1, 2, 2)

sns.kdeplot(data['revenue'])

plt.title('Distribution of revenues(nomal scale)')
data['revenue_log'] = np.log1p(data['revenue']) #changing it to log1

sns.kdeplot(data['revenue_log'])

plt.title('Distribution of revenues in log scale')
fig, ax = plt.subplots(figsize = (8, 10))

data['budget_log'] = np.log1p(data['budget'])



plt.subplot(2,1,1)

plt.scatter(x = data['budget'], y=data['revenue']);plt.ylabel('Revenue'); plt.xlabel('budget')

plt.title('distribution of revenue X budget')

plt.subplot(2,1,2)

plt.scatter(x = data['budget_log'], y=data['revenue']);plt.ylabel('Revenue');plt.xlabel('budget')

plt.title('distribution of revenue X budget(log scale)')

plt.tight_layout()

sns.kdeplot(data['budget_log'])

plt.title('Density distribution of budget(log scale)')

plt.tight_layout()
sns.catplot(x='has_homepage', y='revenue', data=data, aspect=1.5); plt.title('Has homepage X Revenue')

sns.catplot(x='is_part_of_collection', y='revenue', data=data, aspect=1.5); plt.title('Part of colection X Revenue')
data_language = pd.DataFrame(data.groupby('original_language')["revenue"].sum().sort_values(ascending=False).head(10))
data_language #showing only the first 10
data['is_english_original_language'] = 0 #setting all the values to false as default

data.loc[data['original_language'] == 'en', 'is_english_original_language'] = 1
sns.catplot(x='is_english_original_language', y='revenue', data=data)

plt.title('Revenue in relationship to english as original language')
data.drop('original_language', axis=1, inplace=True)
plt.scatter(x = data['popularity'], y=data['revenue'])

plt.ylabel('revenue')

plt.xlabel('popularity')

plt.title('correlation of popularity X revenue')
data[['release_month','release_day','release_year']]=data['release_date'].str.split('/',expand=True).replace(np.nan, -1).astype(int) #getting the month year and day using the string split function and the / as a delimiter; eg: 5/25/2015 -> month 5/ day 25 / year 2015

data.loc[ (data['release_year'] <= 19) & (data['release_year'] < 100), "release_year"] += 2000 ## some rows have 4 digits for the year instead of 2, so the release year < 100 and > 100 is checking that

data.loc[ (data['release_year'] > 19)  & (data['release_year'] < 100), "release_year"] += 1900



releaseDate = pd.to_datetime(data['release_date']) #using the pandas to_datetime function to format the data, get a Series,  and store it in a variable that is gonna be used later to get the day of week and quarter

data['release_dayofweek'] = releaseDate.dt.dayofweek

data['release_quarter'] = releaseDate.dt.quarter
plt.figure(figsize=(20, 8))

plt.scatter(x='release_year', y='revenue', data=data); plt.title('Revenue X year released');
plt.figure(figsize=(15, 8))

sns.countplot(data['release_dayofweek']); plt.title('Days of the week, from 0 - sunday,  to 6 - saturday')
## -1 is the nan values in the original data, that has been replaced by -1

plt.figure(figsize=(15, 8))

sns.countplot(data['release_month']); plt.title('Distribution of movies by month')
plt.figure(figsize=(15, 8))

sns.countplot(data['release_day']); plt.title('Distribution of movies by days in a month')
plt.figure(figsize=(20, 10))

plt.subplot(1, 3, 1)

plt.hist(data['runtime'].fillna(0) / 60, bins=40) #filling the null values with 0

plt.title('Distribution of length of film in hours');

plt.subplot(1, 3, 2)

plt.scatter(data['runtime'].fillna(0), data['revenue'])

plt.title('runtime vs revenue');

plt.subplot(1, 3, 3)

plt.scatter(data['runtime'].fillna(0), data['popularity'])

plt.title('runtime vs popularity');
plt.figure(figsize=(16, 14))

plt.subplot(3, 3, 1)

plt.scatter(data['number_cast'], data['revenue'])

plt.title('Revenue X number of cast members')

plt.subplot(3, 3, 2)

plt.scatter(data['crew_total'], data['revenue'])

plt.title('Revenue X number of crew members')

plt.subplot(3, 3, 3)

plt.scatter(data['num_languages'], data['revenue'])

plt.title('Revenue X number of languages')

plt.tight_layout(1.5)
plt.figure(figsize=(16, 8))

plt.subplot(2, 2, 1)

plt.scatter(data['num_countries'], data['revenue'])

plt.title('Revenue X number of countries')

plt.subplot(2, 2, 2)

plt.scatter(data['keywords_num'], data['revenue'])

plt.title('Revenue X number of keywords')
data['title_length'] = data['original_title'].apply(lambda x: len(x) if x != '' else 0)
plt.figure(figsize=(20, 8))

plt.scatter(data['title_length'], data['revenue']);plt.title('Revenue X title lenght')
corr = data.corr()

corr['revenue'].sort_values(ascending=False).head(20) #the top 20 features positively correlated to our target
corr['revenue'].sort_values(ascending=False).tail(20) #negatively correlated to our target
data.isnull().sum().sort_values(ascending=False)
data.drop(['imdb_id', 'poster_path', 'release_date', 'status','budget_log','cast_character_','language_'], axis=1, inplace=True)
data_dropping_names = data.drop(['original_title','overview','tagline','title'], axis=1)



train = data_dropping_names[data_dropping_names['source'] == 'train'].copy()

test = data_dropping_names[data_dropping_names['source'] == 'test'].copy()



train_labels = train['revenue_log'] #creating labels, our Y_train, gonna use the log as it works better for skewed data



train.drop(['id', 'revenue', 'source', 'revenue_log'], axis=1, inplace=True) # dropping the target and id



test_final = test.drop(['id', 'source','revenue','revenue_log'], axis=1) #this is the final test, the dataset give for our prediction, notice that I am dropping the revenue column here that has been created when we merged the two datasets together
print(train.shape, train_labels.shape,test_final.shape)
train.isnull().sum()
#this pipeline is gonna be use for numerical atributes and standard scaler

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler,RobustScaler, MinMaxScaler



num_pipeline = Pipeline([

        ('imputer', SimpleImputer(strategy="median")),

        ('robust_scaler', RobustScaler()),

        #('minmax_scaler', MinMaxScaler())

    ])
#let's create this function to make it easier and clean to fit the model and use the cross_val_score and obtain results

import time #implementing in this function the time spent on training the model

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import  KFold

from catboost import CatBoostRegressor

import lightgbm as lgb

import xgboost as xgb

import eli5

import gc



n_fold = 10 #number of folds that our function is gonna use and split the training set accordingly

folds = KFold(n_splits=n_fold, shuffle=True, random_state=42)



def train_model(X, X_test, y, params=None, folds=folds, model_type='lgb', plot_feature_importance=False, model=None):



    prediction = np.zeros(X_test.shape[0]) #initializing the prediction matrix with zeros, with the number of training examples in X_test

    scores = [] #this list is gonna be used to store all the scores across different folds

    feature_importance = pd.DataFrame() #initializing this dataframe, it's gonna be used to plot the features importance.

    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):

        print('Fold', fold_n, 'started at', time.ctime())

        if model_type == 'sklearn': #if the model type is sklearn then

            X_train, X_valid = X[train_index], X[valid_index]

        else:

            X_train, X_valid = X.values[train_index], X.values[valid_index]

        y_train, y_valid = y[train_index], y[valid_index]

        

        if model_type == 'lgb': 

            model = lgb.LGBMRegressor(**params, n_estimators = 20000, nthread = 4, n_jobs = -1)

            model.fit(X_train, y_train, 

                    eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='rmse',

                    verbose=1000, early_stopping_rounds=200)

            

            y_pred_valid = model.predict(X_valid)

            y_pred = model.predict(X_test, num_iteration=model.best_iteration_)

            

        if model_type == 'xgb':

            train_data = xgb.DMatrix(data=X_train, label=y_train)

            valid_data = xgb.DMatrix(data=X_valid, label=y_valid)



            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]

            model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200, verbose_eval=500, params=params)

            y_pred_valid = model.predict(xgb.DMatrix(X_valid), ntree_limit=model.best_ntree_limit)

            y_pred = model.predict(xgb.DMatrix(X_test.values), ntree_limit=model.best_ntree_limit)



        if model_type == 'sklearn':

            model = model

            model.fit(X_train, y_train)

            y_pred_valid = model.predict(X_valid).reshape(-1,)

            score = mean_squared_error(y_valid, y_pred_valid)

            

            y_pred = model.predict(X_test)

            

        if model_type == 'cat':

            model = CatBoostRegressor(iterations=20000,  eval_metric='RMSE', **params)

            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True, verbose=False)



            y_pred_valid = model.predict(X_valid)

            y_pred = model.predict(X_test)

        

        scores.append(mean_squared_error(y_valid, y_pred_valid) ** 0.5)

        

        prediction += y_pred #summing all the prediction which is gonna later be divided by the number of folds   

        

        if model_type == 'lgb':

            # feature importance

            fold_importance = pd.DataFrame()

            fold_importance["feature"] = X.columns

            fold_importance["importance"] = model.feature_importances_

            fold_importance["fold"] = fold_n + 1

            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)



    prediction /= n_fold # all the predictions divided by the number of folds(getting the average value)

    

    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

    # Clean up memory

    gc.enable()

    del model, y_pred_valid, X_test,X_train,X_valid, y_pred, y_train

    gc.collect()



    

    if model_type == 'lgb':

        feature_importance["importance"] /= n_fold

        if plot_feature_importance:

            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(

                by="importance", ascending=False)[:50].index



            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]



            plt.figure(figsize=(16, 12));

            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));

            plt.title('LGB Features (avg over folds)');

        

            return scores, prediction, feature_importance 

        return scores, prediction

    else:

        return scores, prediction

    
train_dummies = pd.get_dummies(train)

test_dummies = pd.get_dummies(test)

train_dummies, test_dummies = train_dummies.align(test_dummies, axis=1, join='inner')
print(train_dummies.shape, test_dummies.shape)
########## LGB ########

params = {

          'num_leaves': 30,

         'min_data_in_leaf': 20,

         'objective': 'regression',

         'max_depth': 6,

         'learning_rate': 0.01,

         "boosting": "gbdt",

         "feature_fraction": 0.9,

         "bagging_freq": 1,

         "bagging_fraction": 0.9,

         "bagging_seed": 11,

         "metric": 'rmse',

         "lambda_l1": 0.2,

}

score_lgb, prediction_lgb, _ = train_model(train_dummies, test_dummies, train_labels, params=params, model_type='lgb', plot_feature_importance=True)
score_lgb.sort(reverse=True)

dictvalues = {'RMSE_lgb': score_lgb}
plt.plot(dictvalues['RMSE_lgb'], label = 'RMSE_lgb')

plt.legend()
sub = pd.read_csv('../input/sample_submission.csv')

sub['revenue'] = np.expm1(prediction_lgb)

sub.to_csv("lgb_model.csv", index=False)
xgb_params = {'eta': 0.01,

              'objective': 'reg:linear',

              'max_depth': 7,

              'subsample': 0.8,

              'colsample_bytree': 0.8,

              'eval_metric': 'rmse',

              'seed': 11,

              'silent': True}

score_xgb, prediction_xgb = train_model(train_dummies, test_dummies, train_labels, params=xgb_params, model_type='xgb', plot_feature_importance=True)
score_xgb.sort(reverse=True)

dictvalues.update({'RMSE_XGB': score_xgb})
plt.plot(dictvalues['RMSE_lgb'], label = 'RMSE_lgb')

plt.plot(dictvalues['RMSE_XGB'], label = 'RMSE_XGB')

plt.legend()
sub['revenue'] = np.expm1(prediction_xgb)

sub.to_csv("XGB_model.csv", index=False)
cat_params = {'learning_rate': 0.002,

              'depth': 5,

              'l2_leaf_reg': 10,

              'colsample_bylevel': 0.8,

              'bagging_temperature': 0.2,

              'od_type': 'Iter',

              'od_wait': 100,

              'random_seed': 11,

              'allow_writing_files': False}

score_cat, prediction_cat = train_model(train_dummies, test_dummies, train_labels, params=cat_params, model_type='cat', plot_feature_importance=True)
score_cat.sort(reverse=True)

dictvalues.update({'RMSE_CAT': score_cat})

plt.plot(dictvalues['RMSE_lgb'], label = 'RMSE_lgb')

plt.plot(dictvalues['RMSE_XGB'], label = 'RMSE_XGB')

plt.plot(dictvalues['RMSE_CAT'], label = 'RMSE_CAT')

plt.legend()
sub['revenue'] = np.expm1(prediction_cat)

sub.to_csv("cat_model.csv", index=False)
sub['revenue'] = np.expm1((prediction_lgb + prediction_xgb + prediction_cat) / 3)

sub.to_csv("combined.csv", index=False)
## loading relevant models for this part of the notebook

from keras.models import Sequential, Model

from keras.optimizers import Adam, SGD

from keras.layers import Dense, Activation, Dropout, Input, concatenate,BatchNormalization

from keras.wrappers.scikit_learn import KerasRegressor

from keras.regularizers import l2

from keras.callbacks import EarlyStopping
########### creating a function ir order to make easier to test different models ###############



def build_model(input_shape,n_hidden=10, n_neurons=30, optimizer = SGD(3e-3,momentum=0.9)): # the function has the stochastic gradient descent with momemntum as the default, you might wanna use it and see the results

    model = Sequential()

    options = {"input_shape": input_shape}

    for layer in range(n_hidden): # fixed number of neurons for each hidden layer

        BatchNormalization() #calling batch normalization before the activation...

        model.add(Dense(n_neurons, activation="relu",

                           kernel_regularizer=l2(0.01), **options)) #using the relu activation and L2 regularization, I couldn't make this NN works better and the model didn't generalize well

        options = {}

    BatchNormalization() #adding BN before the output layer

    model.add(Dense(1, **options)) 

    model.compile(loss="mean_squared_error", optimizer=optimizer)

    return model
train_dummies_prepared = num_pipeline.fit_transform(train_dummies)
model_NN = build_model(input_shape= train_dummies_prepared.shape[1:], n_hidden=6,optimizer=Adam(3e-5)) #gonna use 6 hidden layers and Adam 
model_history = model_NN.fit(train_dummies_prepared, train_labels, epochs=2500, batch_size=5, validation_split=0.1,callbacks=[EarlyStopping(patience=100)])
pd.DataFrame(model_history.history).plot(figsize=(8, 5))

plt.grid(True)

plt.gca().set_ylim(0, 50) # set the vertical range to [0-1]