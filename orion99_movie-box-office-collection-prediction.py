# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import ast

from collections import Counter

import datetime

from IPython.display import HTML

import base64



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
dict_columns = ['belongs_to_collection', 'genres', 'production_companies',

                'production_countries', 'spoken_languages', 'Keywords', 'cast', 'crew']



def text_to_dict(df):

    for column in dict_columns:

        df[column] = df[column].apply(lambda x: {} if pd.isna(x) else ast.literal_eval(x))

    return df



text_to_dict(train)

text_to_dict(test)

train.head()
# data fixes from https://www.kaggle.com/somang1418/happy-valentines-day-and-keep-kaggling-3

train.loc[train['id'] == 16,'revenue'] = 192864          # Skinning

train.loc[train['id'] == 90,'budget'] = 30000000         # Sommersby          

train.loc[train['id'] == 118,'budget'] = 60000000        # Wild Hogs

train.loc[train['id'] == 149,'budget'] = 18000000        # Beethoven

train.loc[train['id'] == 313,'revenue'] = 12000000       # The Cookout 

train.loc[train['id'] == 451,'revenue'] = 12000000       # Chasing Liberty

train.loc[train['id'] == 464,'budget'] = 20000000        # Parenthood

train.loc[train['id'] == 470,'budget'] = 13000000        # The Karate Kid, Part II

train.loc[train['id'] == 513,'budget'] = 930000          # From Prada to Nada

train.loc[train['id'] == 797,'budget'] = 8000000         # Welcome to Dongmakgol

train.loc[train['id'] == 819,'budget'] = 90000000        # Alvin and the Chipmunks: The Road Chip

train.loc[train['id'] == 850,'budget'] = 90000000        # Modern Times

train.loc[train['id'] == 1112,'budget'] = 7500000        # An Officer and a Gentleman

train.loc[train['id'] == 1131,'budget'] = 4300000        # Smokey and the Bandit   

train.loc[train['id'] == 1359,'budget'] = 10000000       # Stir Crazy 

train.loc[train['id'] == 1542,'budget'] = 1              # All at Once

train.loc[train['id'] == 1570,'budget'] = 15800000       # Crocodile Dundee II

train.loc[train['id'] == 1571,'budget'] = 4000000        # Lady and the Tramp

train.loc[train['id'] == 1714,'budget'] = 46000000       # The Recruit

train.loc[train['id'] == 1721,'budget'] = 17500000       # Cocoon

train.loc[train['id'] == 1865,'revenue'] = 25000000      # Scooby-Doo 2: Monsters Unleashed

train.loc[train['id'] == 2268,'budget'] = 17500000       # Madea Goes to Jail budget

train.loc[train['id'] == 2491,'revenue'] = 6800000       # Never Talk to Strangers

train.loc[train['id'] == 2602,'budget'] = 31000000       # Mr. Holland's Opus

train.loc[train['id'] == 2612,'budget'] = 15000000       # Field of Dreams

train.loc[train['id'] == 2696,'budget'] = 10000000       # Nurse 3-D

train.loc[train['id'] == 2801,'budget'] = 10000000       # Fracture

test.loc[test['id'] == 3889,'budget'] = 15000000       # Colossal

test.loc[test['id'] == 6733,'budget'] = 5000000        # The Big Sick

test.loc[test['id'] == 3197,'budget'] = 8000000        # High-Rise

test.loc[test['id'] == 6683,'budget'] = 50000000       # The Pink Panther 2

test.loc[test['id'] == 5704,'budget'] = 4300000        # French Connection II

test.loc[test['id'] == 6109,'budget'] = 281756         # Dogtooth

test.loc[test['id'] == 7242,'budget'] = 10000000       # Addams Family Values

test.loc[test['id'] == 7021,'budget'] = 17540562       #  Two Is a Family

test.loc[test['id'] == 5591,'budget'] = 4000000        # The Orphanage

test.loc[test['id'] == 4282,'budget'] = 20000000       # Big Top Pee-wee





# train.loc[train['id'] == 16,'revenue'] = 192864         

# train.loc[train['id'] == 90,'budget'] = 30000000                  

# train.loc[train['id'] == 118,'budget'] = 60000000       

# train.loc[train['id'] == 149,'budget'] = 18000000       

# train.loc[train['id'] == 313,'revenue'] = 12000000       

# train.loc[train['id'] == 451,'revenue'] = 12000000      

# train.loc[train['id'] == 464,'budget'] = 20000000       

# train.loc[train['id'] == 470,'budget'] = 13000000       

# train.loc[train['id'] == 513,'budget'] = 930000         

# train.loc[train['id'] == 797,'budget'] = 8000000        

# train.loc[train['id'] == 819,'budget'] = 90000000       

# train.loc[train['id'] == 850,'budget'] = 90000000       

# train.loc[train['id'] == 1007,'budget'] = 2              

# train.loc[train['id'] == 1112,'budget'] = 7500000       

# train.loc[train['id'] == 1131,'budget'] = 4300000        

# train.loc[train['id'] == 1359,'budget'] = 10000000       

# train.loc[train['id'] == 1542,'budget'] = 1             

# train.loc[train['id'] == 1570,'budget'] = 15800000       

# train.loc[train['id'] == 1571,'budget'] = 4000000        

# train.loc[train['id'] == 1714,'budget'] = 46000000       

# train.loc[train['id'] == 1721,'budget'] = 17500000       

# train.loc[train['id'] == 1865,'revenue'] = 25000000      

# train.loc[train['id'] == 1885,'budget'] = 12             

# train.loc[train['id'] == 2091,'budget'] = 10             

# train.loc[train['id'] == 2268,'budget'] = 17500000       

# train.loc[train['id'] == 2491,'budget'] = 6              

# train.loc[train['id'] == 2602,'budget'] = 31000000       

# train.loc[train['id'] == 2612,'budget'] = 15000000       

# train.loc[train['id'] == 2696,'budget'] = 10000000      

# train.loc[train['id'] == 2801,'budget'] = 10000000       

# train.loc[train['id'] == 335,'budget'] = 2 

# train.loc[train['id'] == 348,'budget'] = 12

# train.loc[train['id'] == 470,'budget'] = 13000000 

# train.loc[train['id'] == 513,'budget'] = 1100000

# train.loc[train['id'] == 640,'budget'] = 6 

# train.loc[train['id'] == 696,'budget'] = 1

# train.loc[train['id'] == 797,'budget'] = 8000000 

# train.loc[train['id'] == 850,'budget'] = 1500000

# train.loc[train['id'] == 1199,'budget'] = 5 

# train.loc[train['id'] == 1282,'budget'] = 9              

# train.loc[train['id'] == 1347,'budget'] = 1

# train.loc[train['id'] == 1755,'budget'] = 2

# train.loc[train['id'] == 1801,'budget'] = 5

# train.loc[train['id'] == 1918,'budget'] = 592 

# train.loc[train['id'] == 2033,'budget'] = 4

# train.loc[train['id'] == 2118,'budget'] = 344 

# train.loc[train['id'] == 2252,'budget'] = 130

# train.loc[train['id'] == 2256,'budget'] = 1 

# train.loc[train['id'] == 2696,'budget'] = 10000000





# test.loc[test['id'] == 3033,'budget'] = 250 

# test.loc[test['id'] == 3051,'budget'] = 50

# test.loc[test['id'] == 3084,'budget'] = 337

# test.loc[test['id'] == 3224,'budget'] = 4  

# test.loc[test['id'] == 3594,'budget'] = 25  

# test.loc[test['id'] == 3619,'budget'] = 500  

# test.loc[test['id'] == 3831,'budget'] = 3  

# test.loc[test['id'] == 3935,'budget'] = 500  

# test.loc[test['id'] == 4049,'budget'] = 995946 

# test.loc[test['id'] == 4424,'budget'] = 3  

# test.loc[test['id'] == 4460,'budget'] = 8  

# test.loc[test['id'] == 4555,'budget'] = 1200000 

# test.loc[test['id'] == 4624,'budget'] = 30 

# test.loc[test['id'] == 4645,'budget'] = 500 

# test.loc[test['id'] == 4709,'budget'] = 450 

# test.loc[test['id'] == 4839,'budget'] = 7

# test.loc[test['id'] == 3125,'budget'] = 25 

# test.loc[test['id'] == 3142,'budget'] = 1

# test.loc[test['id'] == 3201,'budget'] = 450

# test.loc[test['id'] == 3222,'budget'] = 6

# test.loc[test['id'] == 3545,'budget'] = 38

# test.loc[test['id'] == 3670,'budget'] = 18

# test.loc[test['id'] == 3792,'budget'] = 19

# test.loc[test['id'] == 3881,'budget'] = 7

# test.loc[test['id'] == 3969,'budget'] = 400

# test.loc[test['id'] == 4196,'budget'] = 6

# test.loc[test['id'] == 4221,'budget'] = 11

# test.loc[test['id'] == 4222,'budget'] = 500

# test.loc[test['id'] == 4285,'budget'] = 11

# test.loc[test['id'] == 4319,'budget'] = 1

# test.loc[test['id'] == 4639,'budget'] = 10

# test.loc[test['id'] == 4719,'budget'] = 45

# test.loc[test['id'] == 4822,'budget'] = 22

# test.loc[test['id'] == 4829,'budget'] = 20

# test.loc[test['id'] == 4969,'budget'] = 20

# test.loc[test['id'] == 5021,'budget'] = 40 

# test.loc[test['id'] == 5035,'budget'] = 1 

# test.loc[test['id'] == 5063,'budget'] = 14 

# test.loc[test['id'] == 5119,'budget'] = 2 

# test.loc[test['id'] == 5214,'budget'] = 30 

# test.loc[test['id'] == 5221,'budget'] = 50 

# test.loc[test['id'] == 4903,'budget'] = 15

# test.loc[test['id'] == 4983,'budget'] = 3

# test.loc[test['id'] == 5102,'budget'] = 28

# test.loc[test['id'] == 5217,'budget'] = 75

# test.loc[test['id'] == 5224,'budget'] = 3 

# test.loc[test['id'] == 5469,'budget'] = 20 

# test.loc[test['id'] == 5840,'budget'] = 1 

# test.loc[test['id'] == 5960,'budget'] = 30

# test.loc[test['id'] == 6506,'budget'] = 11 

# test.loc[test['id'] == 6553,'budget'] = 280

# test.loc[test['id'] == 6561,'budget'] = 7

# test.loc[test['id'] == 6582,'budget'] = 218

# test.loc[test['id'] == 6638,'budget'] = 5

# test.loc[test['id'] == 6749,'budget'] = 8 

# test.loc[test['id'] == 6759,'budget'] = 50 

# test.loc[test['id'] == 6856,'budget'] = 10

# test.loc[test['id'] == 6858,'budget'] =  100

# test.loc[test['id'] == 6876,'budget'] =  250

# test.loc[test['id'] == 6972,'budget'] = 1

# test.loc[test['id'] == 7079,'budget'] = 8000000

# test.loc[test['id'] == 7150,'budget'] = 118

# test.loc[test['id'] == 6506,'budget'] = 118

# test.loc[test['id'] == 7225,'budget'] = 6

# test.loc[test['id'] == 7231,'budget'] = 85

# test.loc[test['id'] == 5222,'budget'] = 5

# test.loc[test['id'] == 5322,'budget'] = 90

# test.loc[test['id'] == 5350,'budget'] = 70

# test.loc[test['id'] == 5378,'budget'] = 10

# test.loc[test['id'] == 5545,'budget'] = 80

# test.loc[test['id'] == 5810,'budget'] = 8

# test.loc[test['id'] == 5926,'budget'] = 300

# test.loc[test['id'] == 5927,'budget'] = 4

# test.loc[test['id'] == 5986,'budget'] = 1

# test.loc[test['id'] == 6053,'budget'] = 20

# test.loc[test['id'] == 6104,'budget'] = 1

# test.loc[test['id'] == 6130,'budget'] = 30

# test.loc[test['id'] == 6301,'budget'] = 150

# test.loc[test['id'] == 6276,'budget'] = 100

# test.loc[test['id'] == 6473,'budget'] = 100

# test.loc[test['id'] == 6842,'budget'] = 30



power_six = train.id[train.budget > 1000][train.revenue < 100]



for k in power_six :

    train.loc[train['id'] == k,'revenue'] =  train.loc[train['id'] == k,'revenue'] * 1000000

train['has_homepage'] = 0

train.loc[train['homepage'].isnull() == False, 'has_homepage'] = 1



test['has_homepage'] = 0

test.loc[test['homepage'].isnull() == False, 'has_homepage'] = 1



test.head()
plt.scatter(train["has_homepage"], train["revenue"], alpha=.2)
test.loc[test['release_date'].isnull() == True, 'release_date'] = "04/10/99"                                         

train['release_date_mod'] = pd.to_datetime(train['release_date'],format="%m/%d/%y")

train['release_date_mod'] = train['release_date_mod'].mask(train['release_date_mod'].dt.year > 2019, 

                                         train['release_date_mod'] - pd.offsets.DateOffset(years=100))

train['release_date_mod']



test.loc[test['release_date'].isnull() == True, 'release_date'] = "04/10/99"                                         

test['release_date_mod'] = pd.to_datetime(test['release_date'],format="%m/%d/%y")

test['release_date_mod'] = test['release_date_mod'].mask(test['release_date_mod'].dt.year > 2019, 

                                         test['release_date_mod'] - pd.offsets.DateOffset(years=100))

test['release_date_mod']





def make_date_feature(df):

    df["release_year"] = pd.DatetimeIndex(df['release_date_mod']).year

    df["release_day"] = pd.DatetimeIndex(df['release_date_mod']).dayofweek

    

make_date_feature(train)

make_date_feature(test)



train.head()
fig, ax = plt.subplots(figsize = (12, 8))

plt.subplot(2, 2, 1)

plt.scatter(train["release_day"], train["revenue"], alpha=.2)

plt.subplot(2, 2, 2)

plt.scatter(train["release_year"], train["revenue"], alpha=.2)

plt.subplot(2, 2, 3)

plt.hist(train["release_day"])

plt.subplot(2, 2, 4)

plt.hist(test["release_day"])
#plt.hist(train["revenue"])

#plt.hist(np.log1p(train["revenue"]))



#plt.hist(train["budget"])

plt.hist(np.log1p(train["budget"]))



train["log_budget"] = train["budget"].apply(lambda x : np.log1p(x))

train["log_revenue"] = train["revenue"].apply(lambda x : np.log1p(x))
test["log_budget"] = train["budget"].apply(lambda x : np.log1p(x))
all_languages = list(train["original_language"])

Counter(all_languages).most_common() 



train["is_english"] = train["original_language"].apply(lambda x : 1 if x=='en' else 0)

plt.scatter(train["is_english"], train["revenue"], alpha=.2)



test["is_english"] = test["original_language"].apply(lambda x : 1 if x=='en' else 0)
fig, ax = plt.subplots(figsize = (16, 8))

plt.subplot(1, 2, 1)

plt.scatter(train["runtime"], train["revenue"])

plt.subplot(1, 2, 2)

plt.hist(train["runtime"])
plt.scatter(train["popularity"], train["revenue"])
train["in_collection"] = train["belongs_to_collection"].apply(lambda x : 1 if len(x)!=0 else 0)

sns.jointplot(train["in_collection"], train["revenue"], alpha=.5)



test["in_collection"] = test["belongs_to_collection"].apply(lambda x : 1 if len(x)!=0 else 0)
train.head()
def new_features(df):    

    # some features from https://www.kaggle.com/somang1418/happy-valentines-day-and-keep-kaggling-3

    df['budget_year_ratio'] = df['budget'] / (df['release_year'] * df['release_year'])

    df['year_logbudget_ratio'] = df['release_year']/df['log_budget']

    df['popularity_to_mean_year'] = df['popularity'] / df.groupby("release_year")["popularity"].transform('mean')

    df['budget_to_mean_year'] = df['budget'] / df.groupby("release_year")["budget"].transform('mean')

    df['budget_to_runtime'] = df['log_budget'] / df['runtime']

    

    df['budget_popularity_ratio'] = df['log_budget']/df['popularity']

    df['releaseYear_popularity_ratio'] = df['release_year']/df['popularity']

    df['releaseYear_popularity_ratio2'] = df['popularity']/df['release_year']

    return df

    

new_features(train)

new_features(test)
fig, ax = plt.subplots(figsize = (12, 8))

plt.subplot(2, 2, 1)

plt.scatter(train["budget_year_ratio"], train["log_revenue"], alpha=.2)

plt.subplot(2, 2, 2)

plt.scatter(train["year_logbudget_ratio"], train["log_revenue"], alpha=.2)

plt.subplot(2, 2, 3)

plt.scatter(train["popularity_to_mean_year"], train["log_revenue"], alpha=.2)

plt.subplot(2, 2, 4)

plt.scatter(train["budget_to_mean_year"], train["log_revenue"], alpha=.2)
X = train.as_matrix(columns=[

                            "log_budget",

                            "budget_year_ratio",

                            "runtime",

                            "has_homepage",

                            "popularity", 

#                             "popularity_to_mean_year",

#                             "budget_to_mean_year",

                            "release_year",

#                             "budget_to_runtime",

#                             "year_logbudget_ratio",

                            "budget_popularity_ratio",

                            "releaseYear_popularity_ratio",

                            ])

X
Y = train.as_matrix(columns=["log_revenue"])

Y
nan_values = np.argwhere(np.isnan(X))

nan_values
for i in nan_values:

    X[i[0],i[1]]=np.nanmean(X[:,i[1]:i[1]+1])
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=.8, random_state=0)



from xgboost import XGBRegressor

model_xgb = XGBRegressor(

                        gamma=1, 

                        learning_rate=.1,

                        max_depth=3,

                        subsample=1,

                        reg_lambda=1,

                        )

                         

model_xgb.fit(X_train, Y_train,eval_metric='rmse', verbose = True, eval_set = [(X_test, Y_test)])



y_pred = model_xgb.predict(X_test)

y_pred = np.array(np.exp(y_pred)-1)

y_pred



from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, mean_squared_log_error

print(explained_variance_score(np.array(np.exp(Y_test)-1), y_pred))

print(mean_squared_log_error(np.array(np.exp(Y_test)-1), y_pred))
from sklearn.model_selection import KFold



random_seed = 42

k = 10

fold = list(KFold(k, shuffle = True, random_state = random_seed).split(train))

np.random.seed(random_seed)
# from sklearn.ensemble import RandomForestRegressor

# model_rf = RandomForestRegressor(n_estimators=100)

# model_rf.fit(X_train, Y_train)



# y_pred_2 = model_rf.predict(X_test)
# import lightgbm as lgb



# lgb_train = lgb.Dataset(X_train, np.array(Y_train))

# lgb_eval = lgb.Dataset(X_test, np.array(Y_test), reference=lgb_train)



# params = {

#     'boosting_type': 'gbdt',

#     'objective': 'regression',

#     'metric': {'l2', 'l1'},

#     'num_leaves': 31,

#     'learning_rate': 0.05,

#     'feature_fraction': 0.9,

#     'bagging_fraction': 0.8,

#     'bagging_freq': 5,

#     'verbose': 0

# }



# gbm = lgb.train(params,

#                 lgb_train,

#                 num_boost_round=100,

#                 valid_sets=lgb_eval,

#                 early_stopping_rounds=5)
# y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
X_final = test.as_matrix(columns=[

                            "log_budget",

                            "budget_year_ratio",

                            "runtime",

                            "has_homepage",

                            "popularity", 

#                             "popularity_to_mean_year",

#                             "budget_to_mean_year",

                            "release_year",

#                             "budget_to_runtime",

#                             "year_logbudget_ratio",

                            "budget_popularity_ratio",

                            "releaseYear_popularity_ratio",

                            ])



nan_values = np.argwhere(np.isnan(X_final))

for i in nan_values:

    X_final[i[0],i[1]]=np.nanmean(X_final[:,i[1]:i[1]+1])



y_pred_1 = model_xgb.predict(X_final)





results = np.array(np.exp(y_pred_1)-1) 

np.shape(results)
submission = pd.read_csv("../input/sample_submission.csv")

submission["revenue"] = results

submission.head(20)

submission.to_csv("10_sub.csv", index=False)
from IPython.display import HTML

import base64



def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



create_download_link(submission, title = "Download CSV file", filename = "12_sub.csv")