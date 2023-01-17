# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib.pyplot as plt

import seaborn as sns



import ast

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb

import os

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train=pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test.csv")
train.head()
train.describe(include='all')
train.isnull().sum()
print (test.shape,train.shape)
#check number of unique language

train['original_language'].unique()
# Revenue with respect to each language

fig = plt.figure(figsize=(20,10))

sns.barplot('original_language','revenue',data=train)
sns.distplot(train["budget"])
train['Log_budget']=np.log1p(train['budget'])

test['Log_budget']=np.log1p(test['budget'])

sns.distplot(train['Log_budget'])
col=['budget','Log_budget','revenue']

train[col].corr()
def date(df):

    df['release_date'] = pd.to_datetime(df['release_date'])

    df['release_year'] = df['release_date'].dt.year

    df['release_month'] = df['release_date'].dt.month

    df['release_day'] = df['release_date'].dt.day

    df['release_quarter'] = df['release_date'].dt.quarter

    df.drop(columns=['release_date'], inplace=True)

    return df



train=date(train)

test=date(test)

    
fig = plt.figure(figsize=(20,10))

sns.barplot(x="release_day",y="revenue",data=train)

sns.barplot(x="release_month",y="revenue",data=train)
fig = plt.figure(figsize=(20,10))



sns.barplot(x="release_year",y="revenue",data=train)

# we will add genre count for each movies

genres_count=[]

for i in train['genres']:

    if(not(pd.isnull(i))):

        

        genres_count.append(len(eval(i)))

        

    else:

        genres_count.append(0)

train['num_genres'] = genres_count
sns.barplot(x='num_genres', y='revenue', data=train);

plt.title('Revenue for different number of genres in the film');

#genre count for test data

genres_count_test=[]

for i in test['genres']:

    if(not(pd.isnull(i))):

        

        genres_count_test.append(len(eval(i)))

        

    else:

        genres_count_test.append(0)

test['num_genres'] = genres_count_test
#Dropping genre

train.drop(['genres'],axis=1, inplace = True)

test.drop(['genres'],axis=1, inplace = True)

#adding number of production countries

prod_comp_count=[]

for i in train['production_companies']:

    if(not(pd.isnull(i))):

        

        prod_comp_count.append(len(eval(i)))

        

    else:

        prod_comp_count.append(0)

train['num_prod_companies'] = prod_comp_count
sns.barplot(x='num_prod_companies', y='revenue', data=train);

plt.title('Revenue for different number of production companies in the film');

prod_comp_count_test=[]

for i in test['production_companies']:

    if(not(pd.isnull(i))):

        

        prod_comp_count_test.append(len(eval(i)))

        

    else:

        prod_comp_count_test.append(0)

test['num_prod_companies'] = prod_comp_count_test

#Dropping production companies

train.drop(['production_companies'],axis=1, inplace = True)

test.drop(['production_companies'],axis=1, inplace = True)
#Adding num of production countries

prod_coun_count=[]

for i in train['production_countries']:

    if(not(pd.isnull(i))):

        

        prod_coun_count.append(len(eval(i)))

        

    else:

        prod_coun_count.append(0)

train['num_prod_countries'] = prod_coun_count
sns.barplot(x='num_prod_countries', y='revenue', data=train);

plt.title('Revenue for different number of production countries in the film');

prod_coun_count_test=[]

for i in test['production_countries']:

    if(not(pd.isnull(i))):

        

        prod_coun_count_test.append(len(eval(i)))

        

    else:

        prod_coun_count_test.append(0)

test['num_prod_countries'] = prod_coun_count_test
#Dropping production countries

train.drop(['production_countries'],axis=1, inplace = True)

test.drop(['production_countries'],axis=1, inplace = True)
#Here mapping overview represent to 1 anf null to zero

train['overview']=train['overview'].apply(lambda x: 0 if pd.isnull(x) else 1)

test['overview']=test['overview'].apply(lambda x: 0 if pd.isnull(x) else 1)

sns.barplot(x='overview', y='revenue', data=train);

plt.title('Revenue for film with and without overview');

train= train.drop(['overview'],axis=1)

test= test.drop(['overview'],axis=1)
#Addding num of cast in movie

total_cast=[]

for i in train['cast']:

    if(not(pd.isnull(i))):

        

        total_cast.append(len(eval(i)))

        

    else:

        total_cast.append(0)

train['cast_count'] = total_cast
#Visualising through scatter plot for cast number

plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)

plt.scatter(train['cast_count'], train['revenue'])

plt.title('Number of cast members vs revenue');

total_cast=[]

for i in test['cast']:

    if(not(pd.isnull(i))):

        

        total_cast.append(len(eval(i)))

        

    else:

        total_cast.append(0)

test['cast_count'] = total_cast
train= train.drop(['cast'],axis=1)

test= test.drop(['cast'],axis=1)
#Adding num of crews in movies

total_crew=[]

for i in train['crew']:

    if(not(pd.isnull(i))):

        

        total_crew.append(len(eval(i)))

        

    else:

        total_crew.append(0)

train['crew_count'] = total_crew
plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)

plt.scatter(train['crew_count'], train['revenue'])

plt.title('Number of crew members vs revenue');
total_crew=[]

for i in test['crew']:

    if(not(pd.isnull(i))):

        

        total_crew.append(len(eval(i)))

        

    else:

        total_crew.append(0)

test['crew_count'] = total_crew
train= train.drop(['crew'],axis=1)

test= test.drop(['crew'],axis=1)
train= train.drop(['original_title'],axis=1)

test= test.drop(['original_title'],axis=1)
fig = plt.figure(figsize=(20,10))

sns.barplot('original_language','revenue',data=train)
train['original_language'] =train['original_language'].apply(lambda x: 1 if x=='en' else(2 if x=='zh' else 0))

test['original_language'] =test['original_language'].apply(lambda x: 1 if x=='en' else(2 if x=='zh' else 0))
col = ['revenue','budget','popularity','runtime']



plt.subplots(figsize=(10, 8))



corr = train[col].corr()



sns.heatmap(corr, xticklabels=col,yticklabels=col, linewidths=.5,cmap="Blues")
#budget and revenue are highly correlated

sns.regplot(x="budget", y="revenue", data=train)
#status

train.drop(['status'],axis=1,inplace =True)

test.drop(['status'],axis=1,inplace =True)

# We will drop useless column 

train=train.drop(['Keywords'],axis=1)

train=train.drop(['title'],axis=1)

test=test.drop(['Keywords'],axis=1)

test=test.drop(['title'],axis=1)
#tagline 

train['tagline']=train['tagline'].apply(lambda x: 0 if pd.isnull(x) else 1)

test['tagline']=test['tagline'].apply(lambda x: 0 if pd.isnull(x) else 1)

sns.barplot(x='tagline', y='revenue', data=train);

plt.title('Revenue for film with and without tagline');
#runtime has 2 nulls; setting it to the mean

#filling nulls in test

train['runtime']=train['runtime'].fillna(train['runtime'].mean())

test['runtime']=test['runtime'].fillna(test['runtime'].mean())

#adding number of spoken languages for each movie

spoken_count=[]

for i in train['spoken_languages']:

    if(not(pd.isnull(i))):

        

        spoken_count.append(len(eval(i)))

        

    else:

        spoken_count.append(0)

train['spoken_count'] = spoken_count





spoken_count_test=[]

for i in test['spoken_languages']:

    if(not(pd.isnull(i))):

        

        spoken_count_test.append(len(eval(i)))

        

    else:

        spoken_count_test.append(0)

test['spoken_count'] = spoken_count_test
train.drop(['spoken_languages'],axis=1,inplace=True)

test.drop(['spoken_languages'],axis=1,inplace=True)

train.info()


train.isnull().sum()
train.drop(['imdb_id','poster_path'],axis=1,inplace=True)

test.drop(['imdb_id','poster_path'],axis=1,inplace=True)
train=train.drop(['homepage'],axis =1)

test=test.drop(['homepage'],axis =1)
for i, e in enumerate(train['belongs_to_collection'][:5]):

    print(i, e)
#train['belongs_to_collection_ISMISSING']=(train.belongs_to_collection.str.strip()=='').astype(int)

#test['belongs_to_collection_ISMISSING']=(test.belongs_to_collection.str.strip()=='').astype(int)
# Belong to collection contain lots of null values so we will not consider him for model generation

train.drop(['belongs_to_collection'],axis=1,inplace=True)

test.drop(['belongs_to_collection'],axis=1,inplace=True)
train.info()
test.info()
train.head()
train=train.drop(['id'],axis=1)

test=test.drop(['id'],axis=1)

X=train.drop(['revenue'],axis=1)

y=train['revenue']


from sklearn.model_selection import cross_val_score

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)

import math

predicted = y_train.mean()

size = y_test.size

sum =0;

for i in range(size):

    sum = sum + ((y_test.iloc[i] - predicted)*(y_test.iloc[i] - predicted))

mse = sum/size

rmse=math.sqrt(mse)

rmse
#Making First Model

linreg = LinearRegression()

scores = cross_val_score(linreg, X_train, y_train, scoring="neg_mean_squared_error", cv=10)

rmse_scores = np.sqrt(-scores)

print(rmse_scores.mean())
regr = RandomForestRegressor(max_depth=10, min_samples_split=5, random_state=0,n_estimators=100)

scores = cross_val_score(regr, X_train, y_train, scoring="neg_mean_squared_error", cv=10)

rmse_scores = np.sqrt(-scores)

print(rmse_scores.mean())

xgb_model = xgb.XGBRegressor()

scores = cross_val_score(xgb_model, X_train, y_train, scoring="neg_mean_squared_error", cv=10)

rmse_scores = np.sqrt(-scores)

print(rmse_scores.mean())
