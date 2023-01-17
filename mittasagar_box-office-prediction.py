import numpy as np 

import pandas as pd 



import matplotlib.pyplot as plt

import seaborn as sns



import ast

from collections import Counter



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import *



from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

import lightgbm as lgb

import xgboost as xgb
train_data=pd.read_csv("../input/train.csv")

test_data=pd.read_csv("../input/test.csv")
#To get the dimensions of data

train_data.shape, test_data.shape
# Top-5 rows of the dataset

train_data.head()
test_data.head()
# Checking statistical properties of variables

train_data.describe(include='all')
test_data.describe(include='all')
# Checking datatype 

train_data.info()
# Checking and Counting of missing values

train_data.isnull().sum()
train_data.isna().sum().plot(kind="barh", figsize=(20,10))

for i, v in enumerate(train_data.isna().sum()):

    plt.text(v, i, str(v), fontweight='bold', fontsize = 15)

plt.xlabel("Missing Value Count")

plt.ylabel("Features")

plt.title("Missing Value count By Features")
test_data.isnull().sum()
train_data['homepage']=train_data['homepage'].astype(str).apply(lambda x: 1 if x[0:4] == 'http'  else 0)



test_data['homepage']=test_data['homepage'].astype(str).apply(lambda x: 1 if x[0:4] == 'http'  else 0)
# Checking for no.of unique languages

train_data['original_language'].unique(), test_data['original_language'].unique()
#looking for language-wise revenue

plt.figure(figsize=(12,7))

sns.barplot('original_language', 'revenue', data=train_data)

# To convert release date feature into seperate Date, Month, Year features

def date_features(df):

    df['release_date'] = pd.to_datetime(df['release_date'])

    df['release_year'] = df['release_date'].dt.year

    df['release_month'] = df['release_date'].dt.month

    df['release_day'] = df['release_date'].dt.day

    df['release_quarter'] = df['release_date'].dt.quarter

    df.drop(columns=['release_date'], inplace=True)

    return df



train_data=date_features(train_data)

test_data=date_features(test_data)
fig = plt.figure(figsize=(20,10))



# Average revenue by day

plt.subplot(221)

train_data.groupby('release_day').agg('mean')['revenue'].plot(kind='bar')

plt.ylabel('Revenue')

plt.title('Average revenue by day')



# Average revenue by month

plt.subplot(222)

train_data.groupby('release_month').agg('mean')['revenue'].plot(kind='bar')

plt.ylabel('Revenue')

plt.title('Average revenue by month')

loc, labels = plt.xticks()

loc, labels = loc, ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]



plt.show()

#plt.xticks("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
plt.figure(figsize=(15,7))

train_data.groupby('release_year').agg('mean')['revenue'].plot(kind='bar')
train_data['release_year']=np.where(train_data['release_year']> 2019, train_data['release_year']-100, train_data['release_year'])

test_data['release_year']=np.where(test_data['release_year']> 2019, test_data['release_year']-100, test_data['release_year'])
plt.subplots(figsize=(15,10))

sns.heatmap(train_data.corr(),annot=True)
# Plotting budget vs revenue plot

sns.scatterplot('budget', 'revenue',data= train_data)
# Converting Json Format Columns to Dictionary Format



dict_columns = ['belongs_to_collection', 'genres', 'production_companies',

                'production_countries', 'spoken_languages', 'Keywords', 'cast', 'crew']



def text_to_dict(df):

    for column in dict_columns:

        df[column] = df[column].apply(lambda x: {} if pd.isna(x) else ast.literal_eval(x) )

    return df

        

train_data = text_to_dict(train_data)

test_data = text_to_dict(test_data)

train_data['collection_name'] = train_data['belongs_to_collection'].apply(lambda x: x[0]['name'] if x != {} else 0)

train_data.drop('belongs_to_collection', axis=1, inplace=True)





test_data['collection_name'] = test_data['belongs_to_collection'].apply(lambda x: x[0]['name'] if x != {} else 0)

test_data.drop('belongs_to_collection', axis=1, inplace=True)
(train_data['genres'].apply(lambda x: len(x) if x != {} else 0).value_counts().sort_index()).plot(kind='bar',)



for i, v in enumerate(train_data['genres'].apply(lambda x: len(x) if x != {} else 0).value_counts().sort_index()):

    plt.text(i, v, str(v))

    

plt.xlabel('No.of genres in a film')

plt.ylabel('count')
train_data['genres'].apply(lambda x: len(x) if x != {} else 0).value_counts()

list_of_genres = list(train_data['genres'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

most_common_genres=Counter([i for j in list_of_genres for i in j]).most_common(20)

fig = plt.figure(figsize=(10, 6))

data=dict(most_common_genres)

names = list(data.keys())

values = list(data.values())



plt.barh(range(len(data)),values,tick_label=names,color='teal')

plt.xlabel('Count')

plt.title('Movie Genre Count')

plt.show()
train_data['no.of_genres']=train_data['genres'].apply(lambda x: len(x) if x != {} else 0)

train_data['genres'] = train_data['genres'].apply(lambda x: ', '.join(sorted([i['name'] for i in x])) if x != {} else '')



top_genres = [m[0] for m in Counter([i for j in list_of_genres for i in j]).most_common()]

for k in top_genres:

    train_data['genre_' + k] = train_data['genres'].apply(lambda x: 1 if  k in x else 0)

    

    

    

    

test_data['no.of_genres']=test_data['genres'].apply(lambda x: len(x) if x != {} else 0)

test_data['genres'] = test_data['genres'].apply(lambda x: ', '.join(sorted([i['name'] for i in x])) if x != {} else '')



top_genres = [m[0] for m in Counter([i for j in list_of_genres for i in j]).most_common()]

for k in top_genres:

    test_data['genre_' + k] = test_data['genres'].apply(lambda x: 1 if  k in x else 0)    
train_data.groupby('no.of_genres').agg('mean')['revenue'].plot(kind='bar')

plt.ylabel('revenue')

plt.title('Revenue vs #genres')
# Counting the frequency of production company 

list_of_companies = list(train_data['production_companies'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)



most_common_companies=Counter([i for j in list_of_companies for i in j]).most_common(20)



data=dict(most_common_companies)

names = list(data.keys())

values = list(data.values())



fig = plt.figure(figsize=(10, 6))

plt.barh(names,values,color='brown')

plt.xlabel('Count')

plt.title('Top 20 Production Company Count')

plt.show()
# Creating features from production_companies variable



train_data['no.of_production_companies'] = train_data['production_companies'].apply(lambda x: len(x) if x != {} else 0)

train_data['production_companies'] = train_data['production_companies'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')

top_companies = [m[0] for m in Counter([i for j in list_of_companies for i in j]).most_common(30)]

for g in top_companies:

    train_data['production_company_' + g] = train_data['production_companies'].apply(lambda x: 1 if g in x else 0)

    

    

    

    

    

test_data['no.of_production_companies'] = test_data['production_companies'].apply(lambda x: len(x) if x != {} else 0)

test_data['production_companies'] = test_data['production_companies'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')

top_companies = [m[0] for m in Counter([i for j in list_of_companies for i in j]).most_common(30)]

for g in top_companies:

    test_data['production_company_' + g] = test_data['production_companies'].apply(lambda x: 1 if g in x else 0)    
list_of_countries=list(train_data['production_countries'].apply(lambda x:[i['name'] for i in x] if x!={} else []).values)

most_commom_countries=Counter([i for j in list_of_countries for i in j]).most_common(30)

data=dict(most_commom_countries)

names=list(data.keys())

values=list(data.values())



plt.figure(figsize=(15,8))

plt.barh(names, values)

plt.xlabel('count')

plt.title('country-wise movies count ')
train_data['no.of_produc_countries']=train_data['production_countries'].apply(lambda x: len(x) if x!={} else 0)

train_data['production_countries_names']=train_data['production_countries'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x!={} else '')



top_countries = [m[0] for m in Counter([i for j in list_of_countries for i in j]).most_common(30)]

for p in top_countries:

    train_data['produ_country_' + p]=train_data['production_countries_names'].apply(lambda x:1 if p in x else 0)



    

    

    

    

test_data['no.of_produc_countries']=test_data['production_countries'].apply(lambda x: len(x) if x!={} else 0)

test_data['production_countries_names']=test_data['production_countries'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x!={} else '')



top_countries = [m[0] for m in Counter([i for j in list_of_countries for i in j]).most_common(30)]

for p in top_countries:

    test_data['produ_country_' + p]=test_data['production_countries_names'].apply(lambda x:1 if p in x else 0)    
list_of_languages = list(train_data['spoken_languages'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

most_common_languages=Counter([i for j in list_of_languages for i in j]).most_common(20)



train_data['num_languages'] = train_data['spoken_languages'].apply(lambda x: len(x) if x != {} else 0)

train_data['all_languages'] = train_data['spoken_languages'].apply(lambda x: ' '.join(sorted([i['iso_639_1'] for i in x])) if x != {} else '')

top_languages = [m[0] for m in Counter([i for j in list_of_languages for i in j]).most_common(30)]

for g in top_languages:

    train_data['language_' + g] = train_data['all_languages'].apply(lambda x: 1 if g in x else 0)



    

    

    

test_data['num_languages'] = test_data['spoken_languages'].apply(lambda x: len(x) if x != {} else 0)

test_data['all_languages'] = test_data['spoken_languages'].apply(lambda x: ' '.join(sorted([i['iso_639_1'] for i in x])) if x != {} else '')

for g in top_languages:

    test_data['language_' + g] = test_data['all_languages'].apply(lambda x: 1 if g in x else 0)

train_data['runtime'].fillna(train_data['runtime'].mean(), inplace=True)





test_data['runtime'].fillna(test_data['runtime'].mean(), inplace=True)
train=train_data.drop(['id', 'genres','original_language', 'imdb_id', 'original_title', 'overview', 'poster_path', 'production_companies', 'production_countries', 'spoken_languages', 'status', 'tagline', 'title', 'Keywords', 'cast', 'crew', 'production_countries_names',  'all_languages','collection_name',

 'no.of_genres' ], axis=1)



test=test_data.drop(['id', 'genres','original_language', 'imdb_id', 'original_title', 'overview', 'poster_path', 'production_companies', 'production_countries', 'spoken_languages', 'status', 'tagline', 'title', 'Keywords', 'cast', 'crew', 'production_countries_names',  'all_languages','collection_name',

 'no.of_genres' ], axis=1)
train['revenue']=np.log1p(train.revenue)



train['budget']=np.log1p(train.budget)

test['budget']=np.log1p(test.budget)





train['popularity']=np.log1p(train.popularity)

test['popularity']=np.log1p(test.popularity)



train['runtime']=np.log1p(train.runtime)

test['runtime']=np.log1p(test.runtime)
train.shape, test.shape
train_x=train.drop('revenue', axis=1)

train_y=train['revenue']
x_train, x_test, y_train, y_test=train_test_split(train_x, train_y, test_size=0.33)
x_train.shape, x_test.shape, y_train.shape, y_test.shape
model=LinearRegression()

model.fit(x_train, y_train)
predict=model.predict(x_test)

predict_train=model.predict(x_train)
print('RMSE test:', np.sqrt(np.mean((predict - y_test)**2)))

print('RMSE train:', np.sqrt(np.mean((predict_train - y_train)**2)))
model_rf=RandomForestRegressor()

model_rf.fit(x_train, y_train)
predict_rf=model_rf.predict(x_test)

predict_rf_train=model_rf.predict(x_train)
print('Test RMSE RF:', np.sqrt(np.mean((predict_rf - y_test)**2)))

print('Train RMSE RF:', np.sqrt(np.mean((predict_rf_train - y_train)**2)))
Random_Search_Params ={

    'max_features':[1,2,3,4,5,6,7,8,9,10,15,20,25,30,40,50],

    "max_depth": list(range(1,train.shape[1])),

    'n_estimators' : [1, 2, 4, 8, 50, 100,150, 200, 250, 300],

    "min_samples_leaf": [5,10,15,20,25],

    'random_state' : [42] 

    }
random_search = RandomizedSearchCV(

    estimator=RandomForestRegressor(), param_distributions= Random_Search_Params, 

    cv=3,

    refit=True,

    verbose=True)
random_search.fit(x_train, y_train)
random_search.best_params_
model_rf_tune=RandomForestRegressor(random_state=42, 

                                    n_estimators=150, min_samples_leaf=15,

                                    max_features=40, max_depth=86

                                   )
model_rf_tune.fit(x_train, y_train)
predict_rf_tune=model_rf_tune.predict(x_test)



predict_rf_tune_train=model_rf_tune.predict(x_train)
print('Test RMSE RF_tune_:', np.sqrt(np.mean((predict_rf_tune - y_test)**2)))

print('Train RMSE RF_tune:', np.sqrt(np.mean((predict_rf_tune_train - y_train)**2)))
 
params = {'num_leaves': 30,

         'min_data_in_leaf': 20,

         'objective': 'regression',

         'max_depth': 5,

         'learning_rate': 0.01,

         "boosting": "gbrt",

         "metric": 'rmse'}



lgb_model = lgb.LGBMRegressor(**params, n_estimators = 10000, nthread = 4, n_jobs = -1)

lgb_model.fit(x_train, y_train, 

        eval_set=[(x_train, y_train), (x_test, y_test)], eval_metric='rmse',

        verbose=1000, early_stopping_rounds=1000)

Random_Search_lgb_Params ={

    "max_depth": [4,5,6],

    "min_data_in_leaf": [15,20,25],

    'learning_rate': [0.01,0.2,0.3,0.4,0.001,0.002,0.003,0.004,0.005],

    'num_leaves': [25,30,35,40]  }







random_search_lgb = RandomizedSearchCV(

    estimator=lgb_model, param_distributions= Random_Search_lgb_Params, 

    cv=3,

    refit=True,

    random_state=42,

    verbose=True)



random_search_lgb.fit(x_train, y_train)

print('Best score reached: {} with params: {} '.format(random_search_lgb.best_score_, random_search_lgb.best_params_))
tuned_params = {'num_leaves': 35,

         'min_data_in_leaf': 15,

         'objective': 'regression',

         'max_depth': 5,

         'learning_rate': 0.005,

         "boosting": "gbrt",

         "metric": 'rmse'}







lgb_tune_model = lgb.LGBMRegressor(**tuned_params, n_estimators = 10000, nthread = 4, n_jobs = -1)

lgb_tune_model.fit(x_train, y_train, 

        eval_set=[(x_train, y_train), (x_test, y_test)],

        verbose=1000, early_stopping_rounds=1000)
xgb_params = {'eta': 0.01,

              'objective': 'reg:linear',

              'max_depth': 6,

              'min_child_weight': 3,

              'subsample': 0.8,

              

              'eval_metric': 'rmse',

              'seed': 11,

              'silent': True}



model_xgb = xgb.XGBRegressor() 

model_xgb.fit(x_train, y_train)
trainPredict_xgb = model_xgb.predict(x_train)

testPredict_xgb = model_xgb.predict(x_test)
print("xgb test RMSE:", np.sqrt(mean_squared_error(y_test, testPredict_xgb)))

print("xgb train RMSE:", np.sqrt( mean_squared_error(y_train, trainPredict_xgb)))


Random_Search_xgb_Params = {'eta': [0.01,0.02,0.03,0.04,0.05],

              'max_depth': [3,4,5,6,7,8,9,10,11,12,13,14,15,20,25,30,35,40,45,50,60,70,80,90,100],

              'min_child_weight': [3,4,5,6,7,8,9,10,15,20,25],

              'subsample': [0.4,0.5,0.6,0.7,0.8,0.9,1],

              'colsample_bytree': [0.4,0.5,0.6,0.7,0.8,0.9,1],

              }





random_search_xgb = RandomizedSearchCV(

    estimator=xgb.XGBRegressor(), param_distributions= Random_Search_xgb_Params, 

    cv=3,

    refit=True,

    random_state=42,

    verbose=True)



random_search_xgb.fit(x_train, y_train)



random_search_xgb.best_params_
xgb_params = {'eta': 0.04,

              'booster': 'gbtree',

               'max_depth': 8,

              'min_child_weight': 4,

              'subsample': 0.7,

              'colsample_bytree': 0.5,

             'eval_metric': 'rmse'}



model_xgb_tune = xgb.XGBRegressor( params=xgb_params) 

model_xgb_tune.fit(x_train, y_train)



trainPredict_xgb_tune = model_xgb_tune.predict(x_train)

testPredict_xgb_tune = model_xgb_tune.predict(x_test)



print("xgb_tune test RMSE:", np.sqrt(mean_squared_error(y_test, testPredict_xgb_tune)))

print("xgb_tune train RMSE:", np.sqrt( mean_squared_error(y_train, trainPredict_xgb_tune)))
