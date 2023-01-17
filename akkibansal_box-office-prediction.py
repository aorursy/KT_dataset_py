

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import datetime as dt

from wordcloud import WordCloud, STOPWORDS

from collections import OrderedDict



from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score
data = pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test.csv")
'''EDA'''

#check top 5 rows

data.head()
data.info()
#first removing features which are irrelevant for our prediction

data.drop(['imdb_id','poster_path'],axis=1,inplace=True)

test.drop(['imdb_id','poster_path'],axis=1,inplace=True)
#we have a lot of null values for homepage

#Converting homepage as binary

data['has_homepage'] = 0

data.loc[data['homepage'].isnull() == False, 'has_homepage'] = 1

test['has_homepage'] = 0

test.loc[test['homepage'].isnull() == False, 'has_homepage'] = 1



#Homepage v/s Revenue

sns.catplot(x='has_homepage', y='revenue', data=data);

plt.title('Revenue for film with and without homepage');

data=data.drop(['homepage'],axis =1)

test=test.drop(['homepage'],axis =1)


#Converting collections as binary

data['collection'] = 0

data.loc[data['belongs_to_collection'].isnull() == False, 'collection'] = 1

test['collection'] = 0

test.loc[test['belongs_to_collection'].isnull() == False, 'collection'] = 1



#collections v/s Revenue

sns.catplot(x='collection', y='revenue', data=data);

plt.title('Revenue for film with and without collection');

#Collection too increaes the revenue

data=data.drop(['belongs_to_collection'],axis =1)

test=test.drop(['belongs_to_collection'],axis =1)
#Exploring Genres

genres = {}

for i in data['genres']:

    if(not(pd.isnull(i))):

        if (eval(i)[0]['name']) not in genres:

            genres[eval(i)[0]['name']]=1

        else:

                genres[eval(i)[0]['name']]+=1

                

plt.figure(figsize = (12, 8))

#text = ' '.join([i for j in genres for i in j])

wordcloud = WordCloud(background_color="white",width=1000,height=1000, max_words=10,relative_scaling=0.5,normalize_plurals=False).generate_from_frequencies(genres)



plt.imshow(wordcloud)

plt.title('Top genres')

plt.axis("off")

plt.show()

genres = OrderedDict(genres)

#Drama, Comedy and Thriller are popular genres

OrderedDict(sorted(genres.items(), key=lambda t: t[1]))
#adding number of genres for each movie

genres_count=[]

for i in data['genres']:

    if(not(pd.isnull(i))):

        

        genres_count.append(len(eval(i)))

        

    else:

        genres_count.append(0)

data['num_genres'] = genres_count
#Genres v/s revenue

sns.catplot(x='num_genres', y='revenue', data=data);

plt.title('Revenue for different number of genres in the film');
#Adding genres count for test data

genres_count_test=[]

for i in test['genres']:

    if(not(pd.isnull(i))):

        

        genres_count_test.append(len(eval(i)))

        

    else:

        genres_count_test.append(0)

test['num_genres'] = genres_count_test
#Dropping genres

data.drop(['genres'],axis=1, inplace = True)

test.drop(['genres'],axis=1, inplace = True)
#Production companies

#Adding production_companies count for  data

prod_comp_count=[]

for i in data['production_companies']:

    if(not(pd.isnull(i))):

        

        prod_comp_count.append(len(eval(i)))

        

    else:

        prod_comp_count.append(0)

data['num_prod_companies'] = prod_comp_count
#number of prod companies vs revenue

sns.catplot(x='num_prod_companies', y='revenue', data=data);

plt.title('Revenue for different number of production companies in the film');
#Adding production_companies count for  test data

prod_comp_count_test=[]

for i in test['production_companies']:

    if(not(pd.isnull(i))):

        

        prod_comp_count_test.append(len(eval(i)))

        

    else:

        prod_comp_count_test.append(0)

test['num_prod_companies'] = prod_comp_count_test
#number of prod companies vs revenue

sns.catplot(x='num_prod_companies', y='revenue', data=data);

plt.title('Revenue for different number of production companies in the film');
#Dropping production_companies

data.drop(['production_companies'],axis=1, inplace = True)

test.drop(['production_companies'],axis=1, inplace = True)
#production_countries

#Adding production_countries count for  data

prod_coun_count=[]

for i in data['production_countries']:

    if(not(pd.isnull(i))):

        

        prod_coun_count.append(len(eval(i)))

        

    else:

        prod_coun_count.append(0)

data['num_prod_countries'] = prod_coun_count
#number of prod countries vs revenue

sns.catplot(x='num_prod_countries', y='revenue', data=data);

plt.title('Revenue for different number of production countries in the film');
#Adding production_countries count for  test data

prod_coun_count_test=[]

for i in test['production_countries']:

    if(not(pd.isnull(i))):

        

        prod_coun_count_test.append(len(eval(i)))

        

    else:

        prod_coun_count_test.append(0)

test['num_prod_countries'] = prod_coun_count_test
#Dropping production_countries

data.drop(['production_countries'],axis=1, inplace = True)

test.drop(['production_countries'],axis=1, inplace = True)
#handling overview

#mapping overview present to 1 and nulls to 0

data['overview']=data['overview'].apply(lambda x: 0 if pd.isnull(x) else 1)

test['overview']=test['overview'].apply(lambda x: 0 if pd.isnull(x) else 1)

sns.catplot(x='overview', y='revenue', data=data);

plt.title('Revenue for film with and without overview');
data= data.drop(['overview'],axis=1)

test= test.drop(['overview'],axis=1)
#cast

#Adding cast count for  data

total_cast=[]

for i in data['cast']:

    if(not(pd.isnull(i))):

        

        total_cast.append(len(eval(i)))

        

    else:

        total_cast.append(0)

data['cast_count'] = total_cast
plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)

plt.scatter(data['cast_count'], data['revenue'])

plt.title('Number of cast members vs revenue');
#cast

#Adding cast count for  test data

total_cast=[]

for i in test['cast']:

    if(not(pd.isnull(i))):

        

        total_cast.append(len(eval(i)))

        

    else:

        total_cast.append(0)

test['cast_count'] = total_cast
#Dropping cast

data= data.drop(['cast'],axis=1)

test= test.drop(['cast'],axis=1)
#crew

total_crew=[]

for i in data['crew']:

    if(not(pd.isnull(i))):

        

        total_crew.append(len(eval(i)))

        

    else:

        total_crew.append(0)

data['crew_count'] = total_crew
plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)

plt.scatter(data['crew_count'], data['revenue'])

plt.title('Number of crew members vs revenue');
#Adding crew count for  test data

total_crew=[]

for i in test['crew']:

    if(not(pd.isnull(i))):

        

        total_crew.append(len(eval(i)))

        

    else:

        total_crew.append(0)

test['crew_count'] = total_crew
#Dropping crew

data= data.drop(['crew'],axis=1)

test= test.drop(['crew'],axis=1)
#Dropping original_title

data= data.drop(['original_title'],axis=1)

test= test.drop(['original_title'],axis=1)
#How language contributes to revenue

plt.figure(figsize=(15,11)) #figure size



#It's another way to plot our data. using a variable that contains the plot parameters

g1 = sns.boxenplot(x='original_language', y='revenue', 

                   data=data[(data['original_language'].isin((data['original_language'].sort_values().value_counts()[:10].index.values)))])

g1.set_title("Revenue by language", fontsize=20) # title and fontsize

g1.set_xticklabels(g1.get_xticklabels(),rotation=45) # It's the way to rotate the xticks when we use variable to our graphs

g1.set_xlabel('Language', fontsize=18) # Xlabel

g1.set_ylabel('Revenue', fontsize=18) #Ylabel



plt.show()
#Taking only en and zh into consideration as they are the highest grossing

data['original_language'] =data['original_language'].apply(lambda x: 1 if x=='en' else(2 if x=='zh' else 0))

test['original_language'] =test['original_language'].apply(lambda x: 1 if x=='en' else(2 if x=='zh' else 0))
#check correlation between variables

col = ['revenue','budget','popularity','runtime']



plt.subplots(figsize=(10, 8))



corr = data[col].corr()



sns.heatmap(corr, xticklabels=col,yticklabels=col, linewidths=.5, cmap="Reds")
#budget and revenue are highly correlated

sns.regplot(x="budget", y="revenue", data=data)
#Check how revenue depends of day

data['release_date']=pd.to_datetime(data['release_date'])

test['release_date']=pd.to_datetime(data['release_date'])

release_day = data['release_date'].value_counts().sort_index()

release_day_revenue= data.groupby(['release_date'])['revenue'].sum()

release_day_revenue.index=release_day_revenue.index.dayofweek

sns.barplot(release_day_revenue.index,release_day_revenue, data = data,ci=None)

plt.show()
#adding day feature to the data



data['release_day']=data['release_date'].dt.dayofweek 

test['release_day']=test['release_date'].dt.dayofweek 
#filling nulls in test

test['release_day']=test['release_day'].fillna(0)
data.drop(['release_date'],axis=1,inplace=True)

test.drop(['release_date'],axis=1,inplace=True)
#status

print("train data")

print(data['status'].value_counts())

print("test data")

test['status'].value_counts()

#Feature is irrelevant hence dropping

data.drop(['status'],axis=1,inplace =True)

test.drop(['status'],axis=1,inplace =True)
#keywords

Keywords_count=[]

for i in data['Keywords']:

    if(not(pd.isnull(i))):

        

        Keywords_count.append(len(eval(i)))

        

    else:

        Keywords_count.append(0)

data['Keywords_count'] = Keywords_count
#number of prod countries vs revenue

sns.catplot(x='Keywords_count', y='revenue', data=data);

plt.title('Revenue for different number of Keywords in the film');
Keywords_count=[]

for i in test['Keywords']:

    if(not(pd.isnull(i))):

        

        Keywords_count.append(len(eval(i)))

        

    else:

        Keywords_count.append(0)

test['Keywords_count'] = Keywords_count
#Dropping title and keywords

data=data.drop(['Keywords'],axis=1)

data=data.drop(['title'],axis=1)

test=test.drop(['Keywords'],axis=1)

test=test.drop(['title'],axis=1)
#tagline

data['isTaglineNA'] = 0

data.loc[data['tagline'].isnull() == False, 'isTaglineNA'] = 1

test['isTaglineNA'] = 0

test.loc[test['tagline'].isnull() == False, 'isTaglineNA'] = 1



#Homepage v/s Revenue

sns.catplot(x='isTaglineNA', y='revenue', data=data);

plt.title('Revenue for film with and without tagline');

data.drop(['tagline'],axis=1,inplace =True)

test.drop(['tagline'],axis=1,inplace =True)
#runtime has 2 nulls; setting it to the mean

#filling nulls in test

data['runtime']=data['runtime'].fillna(data['runtime'].mean())

test['runtime']=test['runtime'].fillna(test['runtime'].mean())
#spoken languages

#adding number of spoken languages for each movie

spoken_count=[]

for i in data['spoken_languages']:

    if(not(pd.isnull(i))):

        

        spoken_count.append(len(eval(i)))

        

    else:

        spoken_count.append(0)

data['spoken_count'] = spoken_count





spoken_count_test=[]

for i in test['spoken_languages']:

    if(not(pd.isnull(i))):

        

        spoken_count_test.append(len(eval(i)))

        

    else:

        spoken_count_test.append(0)

test['spoken_count'] = spoken_count_test
#dropping spoken_languages

data.drop(['spoken_languages'],axis=1,inplace=True)

test.drop(['spoken_languages'],axis=1,inplace=True)
data.info()
data.head()
data['budget'] = np.log1p(data['budget'])

test['budget'] = np.log1p(test['budget'])
#normalizing budget

#a, b = 1, 100

#m, n = data.budget.min(), data.budget.max()

#data['budget'] = (data.budget - m) / (n - m) * (b - a) + a
y= data['revenue'].values

cols = [col for col in data.columns if col not in ['revenue', 'id']]

X= data[cols].values

y = np.log1p(y)
from sklearn.linear_model import LinearRegression

clf = LinearRegression()

scores = cross_val_score(clf, X, y, scoring="neg_mean_squared_error", cv=10)

rmse_scores = np.sqrt(-scores)

print(rmse_scores.mean())
from sklearn.ensemble import RandomForestRegressor

regr = RandomForestRegressor(max_depth=10, min_samples_split=5, random_state=0,

                             n_estimators=500)

scores = cross_val_score(regr, X, y, scoring="neg_mean_squared_error", cv=10)

rmse_scores = np.sqrt(-scores)

print(rmse_scores.mean())
cols = [col for col in test.columns if col not in ['id']]

X_test= test[cols].values

regr.fit(X,y)

y_pred = regr.predict(X_test)

y_pred=np.expm1(y_pred)

pd.DataFrame({'id': test.id, 'revenue': y_pred}).to_csv('submission_RF.csv', index=False)
import xgboost as xgb

import lightgbm as lgb
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
params = {'num_leaves': 30,

         'min_data_in_leaf': 20,

         'objective': 'regression',

         'max_depth': 5,

         'learning_rate': 0.01,

         "boosting": "gbdt",

         "feature_fraction": 0.9,

         "bagging_freq": 1,

         "bagging_fraction": 0.9,

         "bagging_seed": 11,

         "metric": 'rmse',

         "lambda_l1": 0.2,

         "verbosity": -1}



lgb_model = lgb.LGBMRegressor(**params, n_estimators = 20000, nthread = 4, n_jobs = -1)

lgb_model.fit(X_train, y_train, 

        eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='rmse',

        verbose=1000, early_stopping_rounds=200)



data.head()
lgb_model.fit(X, y, 

        eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='rmse',

        verbose=1000, early_stopping_rounds=200)

y_pred=lgb_model.predict(X_test)

y_pred=np.expm1(y_pred)

pd.DataFrame({'id': test.id, 'revenue': y_pred}).to_csv('submission_LGB.csv', index=False)


xgb_params = {'eta': 0.01,

              'objective': 'reg:linear',

              'max_depth': 7,

              'subsample': 0.8,

              'colsample_bytree': 0.8,

              'eval_metric': 'rmse',

              'seed': 11,

              'silent': True}

xgb_model = xgb.XGBRegressor(**xgb_params, n_estimators = 20000, 

                             nthread = 4, n_jobs = -1)



xgb_model.fit(X, y, 

        eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='rmse',

        verbose=1000, early_stopping_rounds=200)
y_pred = xgb_model.predict(X_test)
y_pred=np.expm1(y_pred)

pd.DataFrame({'id': test.id, 'revenue': y_pred}).to_csv('submission_XGB.csv', index=False)
from sklearn.ensemble import GradientBoostingRegressor



model_gboost = GradientBoostingRegressor()



model_gboost.fit(X_train, y_train)

y_pred = model_gboost.predict(X_test)
y_pred=np.expm1(y_pred)

pd.DataFrame({'id': test.id, 'revenue': y_pred}).to_csv('submission_GradientBoosting.csv', index=False)