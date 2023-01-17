# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import os

import numpy as np

import sklearn
df_train = pd.read_csv('/kaggle/input/1056lab-airbnb-price-predction/train.csv',index_col=0)

df_test = pd.read_csv('/kaggle/input/1056lab-airbnb-price-predction/test.csv',index_col=0)
def check_unique(X, X_test, col):

    return pd.concat([X[col], X_test[col]]).unique()
rep_misssing_val = {"?": np.nan}

df_train.replace(rep_misssing_val, inplace=True)

df_test.replace(rep_misssing_val, inplace=True)
target_col = "price"
import matplotlib.pyplot as plt

import seaborn as sns

fig, axs = plt.subplots(ncols=2, figsize=(20, 5))

sns.distplot(df_train[target_col], ax=axs[0], kde=False, axlabel="Origin")

sns.distplot(np.log1p(df_train[target_col]), kde=False, ax=axs[1], axlabel="Log Transed")

plt.show()
fig, axs = plt.subplots(ncols=2, figsize=(20, 5))

sns.boxplot(df_train[target_col], ax=axs[0])

sns.boxplot(np.log1p(df_train[target_col]), ax=axs[1])

plt.show()
df_train['price'] = np.log(df_train.price+1)
def extract_objects(train, test=None):

    df = pd.concat([train, test], axis=0) if test is not None else train

    

    object_cols = []

    for col in df.columns:

        typ = df[col].dtype

        if typ == object or (typ != int and typ != float):

            object_cols.append(col)

    return object_cols
object_cols = extract_objects(df_train, df_test)

f"Object Columns : {object_cols}"
df_train.info()
df_train.head()
df_train['room_type'].value_counts()
rt_price = df_train.groupby("room_type")["price"].mean()

rt_price
df_train["minimum_nights"].describe()
df_train["neighbourhood_group"].unique()
pd.concat([df_train["neighbourhood_group"], df_test["neighbourhood_group"]]).unique()
#rep_neighbourhood_group = {"Brooklyn":1, "Manhattan": 2, "Queens":3 , "Staten Island": 4,"Bronx": 5}

#df_train["neighbourhood_group"].replace(rep_neighbourhood_group, inplace=True)

#df_test["neighbourhood_group"].replace(rep_neighbourhood_group, inplace=True)
df_test.head()
fig = plt.subplots(figsize = (12,5))

sns.countplot(x = 'room_type', hue = 'neighbourhood_group', data = df_train)
df_train['neighbourhood_group'].value_counts()
df_train["room_type"].unique()
pd.concat([df_train["room_type"], df_test["room_type"]]).unique()
df_test.head()
df_train.info()
object_cols = []

for col, types in zip(df_test.columns, df_test.dtypes):

    if types != "float" and types != "int":

        try:

            df_train[col] = df_train[col].astype(float)

            df_test[col] = df_test[col].astype(float)

        except:

            object_cols.append(col)
df_train.info()
df_train[["host_id", "price"]].groupby(['host_id',], as_index=False).mean().sort_values(by='price', ascending=False)
df_train["host_id"].unique()
df_train[["name", "price"]].groupby(['name'], as_index=False).mean().sort_values(by='price', ascending=False)
#Looking for na values

sns.heatmap(df_train.isnull(), cbar=False)
all_df = pd.concat([df_train.drop('price', axis=1), df_test])

all_df = all_df.replace('?', np.NaN)

columns = all_df.columns

for c in columns:

    all_df[c] = pd.to_numeric(all_df[c], errors='ignore')
columns = all_df.columns

for c in columns:

    if all_df[c].isna().any():

        if all_df[c].dtypes != np.object:

            median = all_df[c].median()

            all_df[c] = all_df[c].replace(np.NaN, median)

        else:

            mfv = all_df[c].mode()[0]

            all_df[c] = all_df[c].replace(np.NaN, mfv)

            

all_df
sns.heatmap(all_df.isnull(), cbar=False)
all_df.info()
all_df['neighbourhood'].value_counts()
all_df['neighbourhood_group'].value_counts()
all_df['neighbourhood_group']= all_df['neighbourhood_group'].astype("category").cat.codes

all_df['neighbourhood'] =all_df['neighbourhood'].astype("category").cat.codes

all_df['room_type'] = all_df['room_type'].astype("category").cat.codes

all_df.info()
all_df= all_df.drop(columns = ['host_id','host_name'])

all_df.info()
all_df["name_length"]=all_df['name'].map(str).apply(len)

print(all_df["name_length"].max())

print(all_df["name_length"].min())

print(all_df["name_length"].idxmax())

print(all_df["name_length"].idxmin())
all_df.at[20661572, 'name']
all_df.at[2539759, 'name']
all_df.info()
print(all_df['neighbourhood'].value_counts()[:5])

neighbourhood = all_df['neighbourhood'].value_counts()[:5].index.tolist()

print(neighbourhood)
all_df['neighbourhood_replaced'] = all_df['neighbourhood']
all_df['neighbourhood_replaced'].value_counts()
hist_mn1=df_train["minimum_nights"][df_train["minimum_nights"]<10].hist()

hist_mn1
all_df["minimum_nights"][all_df["minimum_nights"]>30]
all_df["minimum_nights"][all_df["minimum_nights"]<30]
all_df.loc[(all_df.minimum_nights >30),"minimum_nights"]=30
all_df["minimum_nights"][all_df["minimum_nights"]<=30]
plt.figure(figsize=(10,6))

sns.scatterplot(all_df.longitude,all_df.latitude,hue=all_df.neighbourhood)

plt.ioff()
plt.figure(figsize=(10,6))

sns.scatterplot(all_df.longitude,all_df.latitude,hue=all_df.room_type)

plt.ioff()
all_df.info()
all_df.drop(["name",'last_review'], axis=1, inplace=True)
all_df.info()
fig, axes = plt.subplots(1,1,figsize=(18.5, 6))

sns.distplot(all_df['availability_365'], rug=False, kde=False, color="blue", ax=axes)

axes.set_xlabel('availability_365')

axes.set_xlim(0, 365)
all_df['all_year_avail'] = all_df['availability_365']>353

all_df['low_avail'] = all_df['availability_365']< 12

all_df['no_reviews'] = all_df['reviews_per_month']==0
fig, axes = plt.subplots(1,2, figsize=(21, 6))



sns.distplot(all_df['minimum_nights'], rug=False, kde=False, color="red", ax = axes[0])

axes[0].set_yscale('log')

axes[0].set_xlabel('minimum stay [nights]')

axes[0].set_ylabel('count')



sns.distplot(np.log1p(all_df['minimum_nights']), rug=False, kde=False, color="blue", ax = axes[1])

axes[1].set_yscale('log')

axes[1].set_xlabel('minimum stay [nights]')

axes[1].set_ylabel('count')
fig, axes = plt.subplots(1,2,figsize=(18.5, 6))

sns.distplot(all_df[all_df['reviews_per_month'] < 17.5]['reviews_per_month'], rug=True, kde=False, color="green", ax=axes[0])

#######################################################

sns.distplot(np.sqrt(all_df[all_df['reviews_per_month'] < 17.5]['reviews_per_month']), rug=True, kde=False, color="green", ax=axes[1])

axes[1].set_xlabel('ln(reviews_per_month)')
fig, axes = plt.subplots(1,1, figsize=(21,6))

sns.scatterplot(x= all_df['availability_365'], y=all_df['reviews_per_month'])
all_df['reviews_per_month'] = all_df[all_df['reviews_per_month'] < 30.0]['reviews_per_month']
all_df.info()
categorical_features = all_df.select_dtypes(include=['object'])

print('Categorical features: {}'.format(categorical_features.shape))
all_df['reviews_per_month'] = all_df['reviews_per_month'].fillna(0)
all_df.info()
columns = all_df.columns

for c in columns:

    if all_df[c].dtypes == np.object:

        dummy_df = pd.get_dummies(all_df[[c]])

        all_df = pd.concat([all_df, dummy_df], axis=1)

        all_df = all_df.drop(c, axis=1)

all_df
nrow, ncol = df_train.shape

price_df = df_train[['price']]

df_train = all_df[:nrow]

df_train = pd.concat([df_train, price_df], axis=1)

df_train
nrow, ncol = df_train.shape

df_test= all_df[nrow:]

df_test
from sklearn.model_selection import train_test_split



X = df_train.drop(target_col, axis=1).to_numpy()

y = df_train['price']

X_test = df_test.to_numpy()
X_train, X_val, y_train, y_val = train_test_split(X, y,test_size =0.1, random_state=3)
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import LeaveOneOut

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_squared_log_error

from sklearn.metrics import mean_squared_error
import category_encoders as ce

import xgboost as xgb

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold, StratifiedKFold, RandomizedSearchCV
import category_encoders as ce

import xgboost as xgb

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold, StratifiedKFold, RandomizedSearchCV
X_train, X_val, y_train, y_val = train_test_split(X, y,test_size =0.2, random_state=105)
from sklearn.ensemble import RandomForestRegressor

regrRM1 = RandomForestRegressor(n_estimators=1000)



regrRM1.fit(X_train, y_train)
from sklearn import metrics

print(regrRM1.score(X_train, y_train))

y_pred= regrRM1.predict(X_val)

print(np.sqrt(metrics.mean_squared_error(y_val,y_pred)))
p = regrRM1.predict(X_test)

submit_df_randomrf3_regrRM1= pd.read_csv('/kaggle/input/1056lab-airbnb-price-predction/sampleSubmission.csv')

submit_df_randomrf3_regrRM1['price'] =p

submit_df_randomrf3_regrRM1['price'] =np.exp(p)-1

submit_df_randomrf3_regrRM1.to_csv('submissionrf3_normal.csv', index=False)
import multiprocessing
from sklearn.ensemble import RandomForestRegressor

regrRM1 = RandomForestRegressor(n_estimators=1000,bootstrap= True,max_depth=300,max_features='log2',oob_score= True,random_state=123,n_jobs=multiprocessing.cpu_count()-1)



regrRM1.fit(X_train, y_train)
from sklearn import metrics

print(regrRM1.score(X_train, y_train))

y_pred= regrRM1.predict(X_val)

print(np.sqrt(metrics.mean_squared_error(y_val,y_pred)))
p = regrRM1.predict(X_test)

submit_df_randomrf3_regrRM1= pd.read_csv('/kaggle/input/1056lab-airbnb-price-predction/sampleSubmission.csv')

submit_df_randomrf3_regrRM1['price'] =p

submit_df_randomrf3_regrRM1['price'] =np.exp(p)-1

submit_df_randomrf3_regrRM1.to_csv('submissionrf3_RM1.csv', index=False)
search_params = {

    'n_estimators'      : [5, 10, 20, 30, 50, 100, 300],



    'random_state'      : [2525],

    'n_jobs'            : [1],

    'min_samples_split' : [3,  10,  20,  30,  50, 100],

    'max_depth'         : [3,  10,  20,  30,  50, 100]

}

 

gsr = GridSearchCV(

    RandomForestRegressor(),

    search_params,

    cv = 3,

    n_jobs = multiprocessing.cpu_count()-1,

    verbose=True

)

 

gsr.fit(X_train, y_train)
print(gsr.best_estimator_)
regrRM2 = RandomForestRegressor(max_features='log2', n_estimators=3000,max_depth=300, min_samples_split=.0001,

                      n_jobs=3, random_state=2525)

regrRM2.fit(X_train, y_train)
from sklearn import metrics

print(regrRM2.score(X_train, y_train))

y_pred= regrRM2.predict(X_val)

print(np.sqrt(metrics.mean_squared_error(y_val,y_pred)))
p = regrRM2.predict(X_test)

submit_df_randomrf3_regrRM2= pd.read_csv('/kaggle/input/1056lab-airbnb-price-predction/sampleSubmission.csv')

submit_df_randomrf3_regrRM2['price'] =p

submit_df_randomrf3_regrRM2['price'] =np.exp(p)-1

submit_df_randomrf3_regrRM2.to_csv('submissionrf3_RM2.csv', index=False)
from sklearn.tree import DecisionTreeRegressor

DTree=DecisionTreeRegressor(max_features='log2', max_depth=300,min_samples_leaf=.0001)

DTree.fit(X_train,y_train)

y_predict=DTree.predict(X_val)

from sklearn.metrics import r2_score

r2_score(y_val,y_predict)
from sklearn.tree import DecisionTreeRegressor

DTree=DecisionTreeRegressor()

DTree.fit(X_train,y_train)

y_predict=DTree.predict(X_val)

from sklearn.metrics import r2_score

r2_score(y_val,y_predict)
print(DTree.score(X_train, y_train))

y_pred= DTree.predict(X_val)

print(np.sqrt(metrics.mean_squared_error(y_val,y_pred)))
p = DTree.predict(X_test)

submit_df_DTree= pd.read_csv('/kaggle/input/1056lab-airbnb-price-predction/sampleSubmission.csv')

submit_df_DTree['price'] =p

submit_df_DTree['price'] =np.exp(p)-1

submit_df_DTree.to_csv('submission_DTree.csv', index=False)