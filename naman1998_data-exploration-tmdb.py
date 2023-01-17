import pandas as pd

pd.set_option('display.max_columns', 100)

pd.set_option('display.max_rows', 100)

import numpy as np

import os

import warnings

import matplotlib.pyplot as plt

import seaborn as sns

import ast

import re

import yaml

import ast

from tqdm import tqdm

from datetime import datetime

from scipy import stats

from scipy.stats import norm

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler

import json
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.head(5)
train.iloc[767]["tagline"]
miss = train.isnull().sum()

miss= miss[miss > 0]

miss.sort_values(inplace=True)

miss
# get lengths of text columns

columns = ['original_title', 'title', 'overview', 'tagline']

for col in columns:

    new_col = col + '_len'

    train[new_col] = train[col].apply(lambda x: 0 if x is np.nan else len(x))

    test[new_col] = test[col].apply(lambda x: 0 if x is np.nan else len(x))



# drop ID/URL/text columns

columns.extend(['homepage', 'imdb_id', 'poster_path', 'belongs_to_collection'])



train.drop(columns, axis=1, inplace=True)

test.drop(columns, axis=1, inplace=True)
#Alternate way to drop features using for loop

# train = train[[i for i in train.columns if i not in ["id", "belongs_to_collection", "overview", "poster_path","homepage", "title","original_title","Keywords"]]]    

# test = test[[i for i in test.columns if i not in ["id", "belongs_to_collection", "overview", "poster_path", "tagline","homepage", "title","original_title","Keywords"]]]    
print ('The train data has {0} rows and {1} columns'.format(train.shape[0],train.shape[1]))

print ('The test data has {0} rows and {1} columns'.format(test.shape[0],test.shape[1]))


#Creating Distribution Plot

sns.distplot(train["revenue"])
#Determining Skewness

target= np.power(train['revenue'], 0.169)

print ("The skewness of target is {}".format(target.skew()))

sns.distplot(target)

numeric_data = train.select_dtypes(include=[np.number])

cat_data = train.select_dtypes(exclude=[np.number])

print("There are {} numeric features and {} categorical features".format(numeric_data.shape[1], cat_data.shape[1]))
train.status = train.status.fillna("released")

test.status = test.status.fillna("released")
corr= numeric_data.corr()

sns.heatmap(corr, annot=True, fmt='.2', center=0.0, cmap='coolwarm')
print (corr['revenue'].sort_values(ascending=False)[:], '\n')
cat_data.describe()
print(train[["original_language", "revenue"]].groupby(['original_language'], as_index=False).mean().sort_values(by='revenue', ascending=False))
sp_pivot= train.pivot_table(index='original_language', values='revenue', aggfunc=np.mean).sort_values(by= 'revenue', ascending=False)

sp_pivot
sp_pivot.plot(kind='bar', color='red')
cat = [f for f in train.columns if train.dtypes[f] == 'object']

def anova(frame):

    anv = pd.DataFrame()

    anv['features'] = cat

    pvals = []

    for c in cat:

           samples = []

           for cls in frame[c].unique():

                  s = frame[frame[c] == cls]['revenue'].values

                  samples.append(s)

           pval = stats.f_oneway(*samples)[1]

           pvals.append(pval)

    anv['pval'] = pvals

    return anv.sort_values('pval')



cat_data['revenue'] = train.revenue.values

k = anova(cat_data) 

k['disparity'] = np.log(1./k['pval'].values) 

sns.barplot(data=k, x = 'features', y='disparity') 

plt.xticks(rotation=90) 

plt 
#create numeric plots

num = [f for f in train.columns if train.dtypes[f] != 'object']

nd = pd.melt(train, value_vars = num)

n1 = sns.FacetGrid (nd, col='variable', col_wrap=4, sharex=False, sharey = False)

n1 = n1.map(sns.distplot, 'value')

n1
#Creating Jointplot

sns.jointplot(x=train['runtime'], y=train['revenue'])
d1=dict.fromkeys(set(train["original_language"].values),0 )

for i in train["original_language"].values:

    d1[i]+=1
train.loc[train.release_date.isnull(), 'release_date'] = train.release_date.mode()[0]

test.loc[test.release_date.isnull(), 'release_date'] = test.release_date.mode()[0]
train["runtime"].fillna(train["runtime"].mean, inplace=True)

test["runtime"].fillna(test["runtime"].mean, inplace=True)



train["spoken_languages"].fillna("en", inplace=True)

test["spoken_languages"].fillna("en", inplace=True)



train = train.drop(["Keywords"], axis = 1)

test = test.drop(["Keywords"], axis = 1)
def expand_release_date(df):

    df.release_date = pd.to_datetime(df.release_date)



    df['release_year'] = df.release_date.dt.year

    df['release_year'] = df.release_year.apply(lambda x: x-100 if x > 2020 else x)

    

    df['release_month'] = df.release_date.dt.month

    df['release_day'] = df.release_date.dt.dayofweek

    df['release_quarter'] = df.release_date.dt.quarter

    

    return df



train = expand_release_date(train)

test = expand_release_date(test)
train["genres"].fillna("[{'id': 18, 'name': 'Drama'}]", inplace=True)

test["genres"].fillna("[{'id': 18, 'name': 'Drama'}]", inplace=True)

train["production_countries"].fillna("[{'iso_3166_1': 'US', 'name': 'United States of America'}]"

, inplace=True)

test["production_countries"].fillna("[{'iso_3166_1': 'US', 'name': 'United States of America'}]"

, inplace=True)

train["production_companies"].fillna("[{'name': 'Paramount Pictures', 'id': 4}]"

, inplace=True)

test["production_companies"].fillna("[{'name': 'Paramount Pictures', 'id': 4}]"

, inplace=True)
print('Number of genres in films')

train['genres'].apply(lambda x: len(x) if x != {} else 0).value_counts()
def conj(j):

    return [i["id"] for i in eval(j)]

train["genres"]=train["genres"].apply(lambda l: conj(l))

test["genres"]=test["genres"].apply(lambda l: conj(l))
def conj(j):

    return [i["name"] for i in eval(j)]

train["production_companies"]=train["production_companies"].apply(lambda l: conj(l))

test["production_companies"]=test["production_companies"].apply(lambda l: conj(l))
def conj(j):

    return [i["iso_3166_1"] for i in eval(j)]

train["production_countries"]=train["production_countries"].apply(lambda l: conj(l))

test["production_countries"]=test["production_countries"].apply(lambda l: conj(l))
def conj(j):

    if pd.isnull(j):

        return np.nan

    return [i["name"] for i in eval(str(j))]

train["cast"]=train["cast"].apply(lambda l: conj(l))

test["cast"]=test["cast"].apply(lambda l: conj(l))
def conj(j):

    if pd.isnull(j):

        return np.nan

    return [i["name"] for i in eval(str(j))]

train["crew"]=train["crew"].apply(lambda l: conj(l))

test["crew"]=test["crew"].apply(lambda l: conj(l))
from collections import Counter

list1= list()

for i in train.dropna().cast:

    if i:

        list1.extend(i)
from collections import Counter

tlist1= list()

for i in test.dropna().cast:

    if i:

        tlist1.extend(i)
# q = Counter([i[1]for i in Counter(list1).most_common()]).most_common()
train=train.dropna(axis=0)

train.cast = train.cast.apply(lambda y: np.nan if len(y)==0 else y)
test=test.dropna(axis=0)

test.cast = test.cast.apply(lambda y: np.nan if len(y)==0 else y)
train=train.dropna(axis=0)

test=test.dropna(axis=0)
list2=[]

for i,j in zip(train["revenue"],train["cast"]):

    for k in range(len(j)):

        list2.append(i)
# tyu=[len(i) for i in train["cast"]]

# tyu.index(0)
app_count = dict(Counter(list1))

list3=list(app_count.values())

rate=dict([(k,i/j) for i,j,k in zip(list2,list3,app_count.keys())])

def sume(l):

    sum1=0

    for i in l:

        sum1+=rate[i]

    return sum1/len(l)

train["cast_rev"]=train["cast"].apply(lambda l : sume(l))
col= [ "id","original_language", "spoken_languages", "title_len", "runtime"]

for i in col:

    train.drop(i, axis=1, inplace=True)
col= [ "id","original_language", "spoken_languages", "title_len", "runtime"]

for i in col:

    test.drop(i, axis=1, inplace=True)
# lo=list(set(np.hstack(train['production_companies'])))

# enco=[]

# for i in train["production_companies"]:

#     if type(i)== list:

#         en=np.zeros((len(lo),),dtype="int")

#         for j in i:

#             en[lo.index(j)]=1

#         enco.append(list(en))

#     else :

#         enco.append(i)



train["cast_rev"]/=min(train["cast_rev"])

# test["cast_rev"]/=min(test["cast_rev"])
train["cast"]
train["cast_no"]=train["cast"].apply(lambda l: np.max([app_count[i] for i in l]))
test["cast"]
# test["cast_no"]=test["cast"].apply(lambda l: np.max([app_count[i] for i in l]))
train=train.drop("cast",axis=1)

test=test.drop("cast",axis=1)
#This feature came out to be of less significance, therefore no need for doin this 

# lo1=list(set(np.hstack(train['genres'])))

# enco1=[]

# for i in train["genres"]:

#     if type(i)== list:

#         en=np.zeros((len(lo1),),dtype="int")

#         for j in i:

#             en[lo1.index(j)]=1

#         enco1.append(list(en))

#     else :

#         print(type(i))

#         enco1.append(i)
# train[["g"+str(i) for i in range(20)]]=pd.DataFrame(enco1,index=train.index)
train.drop("crew",axis=1,inplace=True)

test.drop("crew",axis=1,inplace=True)
train.drop("production_countries",axis=1,inplace=True)

test.drop("production_countries",axis=1,inplace=True)
train.drop(["genres","release_date"],axis=1,inplace=True)

test.drop(["genres","release_date"],axis=1,inplace=True)
from collections import Counter

list1= list()

for i in train.dropna().production_companies:

    if i:

        list1.extend(i)

train.production_companies = train.production_companies.apply(lambda y: np.nan if len(y)==0 else y)

train=train.dropna(axis=0)
from collections import Counter

tlist1= list()

for i in test.dropna().production_companies:

    if i:

        tlist1.extend(i)

test.production_companies = test.production_companies.apply(lambda y: np.nan if len(y)==0 else y)

test=test.dropna(axis=0)
app_count = dict(Counter(list1))
train["production_no"]=train["production_companies"].apply(lambda l: np.max([app_count[i] for i in l]))

# test["production_no"]=test["production_companies"].apply(lambda l: np.max([app_count[i] for i in l]))
train.drop("production_companies",axis=1,inplace=True)

test.drop("production_companies",axis=1,inplace=True)
ct = OneHotEncoder()

train["status"]=ct.fit_transform([train["status"]]).toarray()[0]

test["status"]=ct.fit_transform([test["status"]]).toarray()[0]
train.drop("status",axis=1,inplace=True)

test.drop("status",axis=1,inplace=True)
sc = StandardScaler()

scaled = sc.fit_transform(train.values)

df1=pd.DataFrame(scaled)

X = df1[[i for i in df1.columns if i != 2]].values

y = df1[2].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/4, random_state=0)
from sklearn.linear_model import LinearRegression

regressor= LinearRegression()

regressor.fit(X_train, y_train)

y_pred1= regressor.predict(X_test)
def rmse(pred, actual):

    return np.sqrt(((pred-actual)**2).mean())
print("%.20f"%rmse(y_pred1, y_test))
from sklearn.metrics import r2_score

print("%.10f"%r2_score(y_test, y_pred1))
from sklearn.ensemble import RandomForestRegressor 

regressor2= RandomForestRegressor(n_estimators= 400)

regressor2.fit(X_train, y_train)

y_pred2= regressor2.predict(X_test)
r2_score(y_test, y_pred2)
rmse(y_pred2, y_test)