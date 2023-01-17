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
import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import MinMaxScaler

from xgboost import XGBRegressor

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV, cross_val_score

from scipy.stats import skew

from sklearn.compose import ColumnTransformer

import time

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import r2_score
train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/train.csv')

test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/test.csv')
train.head()
test.head()
train.describe().T
train.shape
test.shape
train.isnull().sum()
is_null = train.isnull().sum()

is_null.sort_values(inplace=True)

is_null.plot.bar()

plt.figure(figsize=(12, 6))

plt.show()
top_country_population = train.Population.groupby(train['Country_Region']).max().sort_values(ascending=False)[:10]

new_df = pd.DataFrame(top_country_population).reset_index()

plt.figure(figsize=(10, 8))

sns.barplot(x = 'Country_Region', y = 'Population', data=new_df)

plt.title("The top 10 countries with Population")

plt.show()
top_country_population
df = train[train['Target']=='ConfirmedCases']

top_with_confirmed = df.TargetValue.groupby(train['Country_Region']).sum().sort_values(ascending=False)[:10]

new_df = pd.DataFrame(top_with_confirmed).reset_index()

plt.figure(figsize=(10, 8))

sns.barplot(x = 'Country_Region', y = 'TargetValue', data=new_df)

plt.title("The top 10 countries with Confirmed")

plt.show()
train.dtypes
train['Date'] = pd.to_datetime(train['Date'])
test['Date'] = pd.to_datetime(test['Date'])
cat_cols = train.select_dtypes(include='object')

cat_cols.columns
num_cols = train.select_dtypes(exclude=['object','datetime64[ns]'])

num_cols.columns
train.County.nunique()
train.Province_State.nunique()
train.County.unique()[:50]
train.Province_State.unique()[:50]
train.Country_Region.nunique()
#Univariate Analysis

fig = plt.figure(figsize=(10, 6))

for i in range(len(num_cols.columns)):

    fig.add_subplot(2,2,i+1)

    sns.boxplot(y=num_cols.iloc[:,i])

plt.tight_layout()

plt.show()
#Bivariate Analysis

fig = plt.figure(figsize=(10, 6))

for i in range(len(num_cols.columns)):

    fig.add_subplot(2,2,i+1)

    sns.scatterplot(num_cols.iloc[:,i],train['TargetValue'])

plt.tight_layout()

plt.show()
#Distibution of Data

fig = plt.figure(figsize=(10, 6))

for i in range(len(num_cols.columns)):

    fig.add_subplot(2,2,i+1)

    sns.distplot(num_cols.iloc[:,i].dropna() , rug = True, hist = True,

                label = 'UW', kde_kws = {'bw':0.1}, color = 'green')

    plt.xlabel(num_cols.columns[i])

plt.tight_layout()

plt.show()
# figure, (ax1, ax2) = plt.subplots(ncols=1, nrows=2)

# figure.set_size_inches(15, 20)



# _ = sns.regplot(train['Population'], train['TargetValue'], ax=ax1)

# _ = sns.regplot(train['Weight'], train['TargetValue'], ax=ax2)
num_correlations = train.select_dtypes(exclude=['object']).corr()

plt.figure(figsize=(5, 5))

sns.heatmap(num_correlations>.8, square = True)
for df in [train, test]:

    for col in ('County', 'Province_State'):

        df[col].fillna('None', inplace=True)
encoder = OneHotEncoder()



temp = pd.DataFrame(encoder.fit_transform(train[['Target']]).toarray(), columns=

                   ['ConfirmedCases', 'Fatalities'])

train = train.join(temp)
temp_2 = pd.DataFrame(encoder.fit_transform(test[['Target']]).toarray(), columns=

                   ['ConfirmedCases', 'Fatalities'])

test = test.join(temp_2)
train1= train[train['Target']=='ConfirmedCases']

data1 = pd.DataFrame()

data1['values'] =train1.TargetValue.groupby(train1['Country_Region']).sum().sort_values(ascending= False)

data1['country'] = data1.index
k = len(data1['country'])

dict1 = {}

for i in data1['country']:

    dict1[i] = k

    k =k-1
list1=[]

train['encoded_country']=0

for i in train['Country_Region']:

    list1.append(dict1[i])

train['encoded_country'] = list1
list1=[]

test['encoded_country']=0

for i in test['Country_Region']:

    list1.append(dict1[i])

test['encoded_country'] = list1
list_2 = []

train['month'] = 0

for i in train['Date']:

    list_2.append(i.month)

train['month'] = list_2
list_2 = []

test['month'] = 0

for i in test['Date']:

    list_2.append(i.month)

test['month'] = list_2
list_2 = []

train['day'] = 0

for i in train['Date']:

    list_2.append(i.day)

train['day'] = list_2
list_2 = []

test['day'] = 0

for i in test['Date']:

    list_2.append(i.day)

test['day'] = list_2
train.drop('Date', axis = 1, inplace=True)

test.drop('Date', axis = 1, inplace=True)
test.drop(['Country_Region' , 'Target'] , axis = 1, inplace=True)

train.drop(['Country_Region' , 'Target'] , axis = 1, inplace=True)
test.drop(['County' , 'Province_State'] , axis = 1, inplace=True)

train.drop(['County' , 'Province_State'] , axis = 1, inplace=True)
X = train.drop(['TargetValue'], axis=1)

y = train['TargetValue'].copy()
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=.2, train_size=.8, random_state=0)
# ts = time.time() 



# model = XGBRegressor(learning_rate =0.01, n_estimators = 800, max_depth=10,

#                     min_child_weight=0, gamma=0, subsample=.7, colsample_bytree=.7,

#                     objective='reg:squarederror', nthread=-1, scale_pos_weight=1,

#                     seed=27, reg_alpha=0.00006, n_jobs=-1)



# model.fit(X_train, y_train)

# pred_1=model.predict(X_valid)

# print(time.time() - ts)
# print(mean_absolute_error(y_valid, pred_1))
ts = time.time()

model_1 = RandomForestRegressor(random_state=7)



model_1.fit(X_train, y_train)

preidct_2 = model_1.predict(X_valid)

print(mean_absolute_error(preidct_2, y_valid))
# print(r2_score(y_valid, preidct_2))
# print(r2_score(y_valid, pred_1))
test = test.rename(columns={'ForecastId' : 'Id'})
predictions = model_1.predict(test)
t =pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/test.csv')

sub = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/submission.csv')

output = pd.DataFrame({'Id' : t.ForecastId, 

                      'TargetValue': predictions})
first = output.groupby(['Id'])['TargetValue'].quantile(q=0.05).reset_index()

mid = output.groupby(['Id'])['TargetValue'].quantile(q=0.5).reset_index()

third = output.groupby(['Id'])['TargetValue'].quantile(q=0.95).reset_index()
first.columns=['Id','q0.05']

mid.columns=['Id','q0.5']

third.columns=['Id','q0.95']
first = pd.concat([first, mid['q0.5'], third['q0.95']], 1)

first['q0.05']=first['q0.05']

first['q0.5']=first['q0.5']

first['q0.95']=first['q0.95']
sub=pd.melt(first, id_vars=['Id'], value_vars=['q0.05','q0.5','q0.95'])
sub['variable']=sub['variable'].str.replace("q","", regex=False)

sub['ForecastId_Quantile']=sub['Id'].astype(str)+'_'+sub['variable']

sub['TargetValue']=sub['value']
sub=sub[['ForecastId_Quantile','TargetValue']]

sub.reset_index(drop=True,inplace=True)

sub.to_csv("submission.csv",index=False)

sub.head()