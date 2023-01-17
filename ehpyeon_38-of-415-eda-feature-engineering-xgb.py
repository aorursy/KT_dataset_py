# Analyzing

import pandas as pd

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import minmax_scale

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans

from scipy import stats

import datetime

import math



# Visualizing

import matplotlib.pyplot as plt

import seaborn as sns

import missingno as msno

import folium

from folium import plugins

#import geojson

from shapely.geometry import shape, Point, multipolygon

#from shap import TreeExplainer, summary_plot

%matplotlib inline

import matplotlib.ticker as ticker



# Others

import warnings 

import gc

warnings.filterwarnings('ignore')



# Modeling

import xgboost as xgb
## Save to Var

train = pd.read_csv('../input/train.csv')

test  = pd.read_csv('../input/test.csv')

all = pd.concat([train,test],axis=0)

## Checking

print("1.train Shape: ",train.shape)

print("2.test Shape: ",test.shape)

print("3.all Shape: ",all.shape,"\n")
def eda_features(data) : 

  checked = pd.DataFrame.from_records([(col, data[col].dtype, data[col].count(),data[data[col]==0][col].count(),data[data[col]!=0][col].min(),data[data[col]!=0][col].max(),data[col].isnull().sum(),data[col].nunique(), data[col].unique()[:5]) for col in data.columns],

                          columns=['Column_Name', 'Data_Type','Counts','0#','Min(0x)','Max(0x)','Nulls','Nunique','Val_Unique']).sort_values(by=['Nunique'])

  return checked



## add skewness and kurtosis
def cutting(data,col,n) : 

  data[col] = pd.cut(data[col],n)

  data = data

  return data
def counting_plot(data, count_info, n):

  rows = math.ceil(len(count_info)/3)

  fig, axes = plt.subplots(rows,3, figsize=(20, rows*4))

  for col in count_info:

    if data[col].nunique() > n:

      data = cutting(data, col, n)

    r = count_info.index(col)//3

    c = count_info.index(col)%3

    plot = sns.countplot(data=data,x=col,ax=axes[r,c])

def relation_box(data, count_info, n):

  rows = math.ceil(len(count_info)/3)

  fig, axes = plt.subplots(rows,3, figsize=(20, rows*4))

  data = data[data['price'].isnull()==False]

  for col in count_info:

    r = count_info.index(col)//3

    c = count_info.index(col)%3

    if data[col].nunique() > n: 

      data = cutting(data, col, n)

    sns.boxplot(x=col, y='price', data=data ,ax=axes[r,c])

  plt.show()

    
def relation_bar(data, count_info, n):

  rows = math.ceil(len(count_info)/3)

  fig, axes = plt.subplots(rows,3, figsize=(20, rows*4))

  data = data[data['price'].isnull()==False]

  for col in count_info:

    r = count_info.index(col)//3

    c = count_info.index(col)%3

    if data[col].nunique() > n: 

      data = cutting(data, col, n)

    sns.barplot(x=col, y='price',  data=data ,ax=axes[r,c])

  plt.show()
##combination

def combination_chart(data, count_info, n):

  rows = len(count_info)

  fig, axes = plt.subplots(rows,4, figsize=(24, rows*3.7))

  data = data[data['price'].isnull()==False]

  for col in count_info:

    r = count_info.index(col)

    if data[col].dtype in ['int64','float64'] :

      sns.regplot(x=col, y='price', data=data, ax=axes[r,3])

    if data[col].nunique() > n: 

      data = cutting(data, col, n)

    sns.countplot(data=data,x=col,ax=axes[r,0])

    if col != 'price':

      sns.barplot(x=col, y='price', data=data ,ax=axes[r,1])

      sns.boxenplot(x=col, y='price', data=data ,ax=axes[r,2])

  plt.show()

  

  

  

 
## step0. Pricehist

def pricehist(data, x, type):

    data = data[data['price'].isnull()==False] #If the dataset is concated with train and test datas, we sholud get rid of test data which don't have target feature('price')

    if data[x].nunique() > 30:

        apple = data.groupby(pd.cut(data[x],30))['price'].mean().sort_index().plot(kind=type,color=(0.2,0.4,0.6,0.6))

    else :

        apple = data.groupby([x])['price'].mean().sort_index().plot(kind=type,color=(0.2,0.4,0.6,0.6))

    return apple

  
## step1. Logarize = Independent var or Dependent var it-self.

def logarize(data,*variables):

  for x in variables :

    data['log_{}'.format(x)] = np.log(data[x])

    data.drop(x,axis=1,inplace=True)

  return data
### step1. Delete

def delete(data, *variables):

  for var in variables :

    del data[var]

  return data
## step2. Categorize

def categorize(data,*variables):

  for x in variables:

    data[x] = data[x].astype('category')

    data = pd.concat([data,pd.get_dummies(data[x],prefix=x,drop_first=True)], axis=1)

    data.drop([x], axis=1, inplace = True)

  return data
## step2. Binarize = 0=>0, other=>1

def binarize(data,*variables):

  for x in variables :

    data['{}_binarized'.format(x)] = [(0 if x== 0 else 1) for x in data[x]]

    data.drop([x], axis=1, inplace = True)

  return data
## step2. Standardize = 

def standardize(data, *variables):

  for x in variables:

    data['{}_standardized'.format(x)] = minmax_scale(data[x])

    data.drop([x],axis=1,inplace=True)

  return data
## step3. Cutagorize : alternative to Categorize, cut and categorize to infinitive data 

def cutagorize(data,n,*variables) :

  tn_list = list(map(lambda x:str(x+1), range(n)))

  for var in variables : 

    data['{}_cut'.format(var)] = pd.qcut(data[var],n,labels = tn_list)

    data = categorize(data,'{}_cut'.format(var))

    data.drop([var],axis=1,inplace=True)

  return data
# Date

## Divide date values 

def divide_date(x):

  return x[:8]



## divide year values

def divide_year(x):

  return x[:4]



## divide month values

def divide_month(x):

  return x[4:6]



## divide day values

def divide_day(x):

  return x[6:8]



## divide year and month values

def divide_yearmonth(x):

  return x[:6]



## transfer the data str to datetime

def todatetime(x):

  return datetime.datetime.strptime(divide_date(x),'%Y%m%d')
## toDatetime

def date_maker(data,dt):

  data[dt] = data[dt].apply(lambda x : todatetime(x))

  return data
def year_maker(data,dt):

  data['year'] = data[dt].apply(lambda x : divide_year(x))

  return data
def month_maker(data,dt):

  data['month'] = data[dt].apply(lambda x : divide_month(x))

  return data
def yearmonth_maker(data,dt):

  data['yearmonth'] = data[dt].apply(lambda x : divide_yearmonth(x))

  return data
# Yr_built vs Yr_renovated

## returning the most recent year

def bigger(data,a,b):

  results = []

  if data[a] >= data[b]:

    results.append(data[a])

  else :

    results.append(data[b])

  return results
def bigger_select(data,a,b):

  results = []

  for i in range(len(data[a])):

    if data[a][i] >= data[b][i]:

      results.append(data[a][i])

    else :

      results.append(data[b][i])

  return results

  
#bigger(train,"yr_built","yr_renovated")

train.columns
train = pd.read_csv('../input/train.csv')

train.head(10)
train.head()
eda_features(train)
train = pd.read_csv('../input/train.csv')

train = yearmonth_maker(train,'date')

date_maker(train,'date')

col_list = ['date','yearmonth','bedrooms','bathrooms','waterfront','floors','view','condition','grade','sqft_living','sqft_lot','sqft_above','sqft_basement','sqft_living15','sqft_lot15','yr_built','yr_renovated','lat','long','zipcode']

combination_chart(train,col_list,30)
corrMatt = train[train.columns]

corrMatt = corrMatt.corr()

mask = np.zeros_like(corrMatt)

mask[np.triu_indices_from(mask)] = True
corrMatt['price'].sort_values(axis=0, ascending=False)
fig, axes = plt.subplots(figsize=(14,12))

sns.heatmap(corrMatt, mask=mask, square=True, annot=True, cmap='Greens')

axes.set_title('Correlation Matters', fontsize = 20)

plt.show()
fig, axes = plt.subplots(1,2, figsize=(10, 4))



sns.regplot(data=train, x='long', y='lat',

           fit_reg=False,

           scatter_kws={'s': 10},

           ax=axes[0])

axes[0].grid(False)

axes[0].set_title('Location of train data', fontsize=15)



sns.regplot(data=test, x='long', y='lat',

           fit_reg=False,

           scatter_kws={'s': 10},

           ax=axes[1])

axes[1].grid(False)

axes[1].set_title('Location of test data', fontsize=15)



plt.show()
map_count = folium.Map(location=[train['lat'].mean(), train['long'].mean()],

                      min_zoom=8,

                      #max_zoom=11,

                      width=660,  # map size scaling

                      height=440)



lat_long_data = train[['lat', 'long']].values.tolist()

h_cluster = folium.plugins.FastMarkerCluster(lat_long_data).add_to(map_count)



map_count
## Save to Var

train = pd.read_csv('../input/train.csv')

test  = pd.read_csv('../input/test.csv')

all = pd.concat([train,test],axis=0, ignore_index=True)

## Checking

print("1.train Shape: ",train.shape)

print("2.test Shape: ",test.shape)

print("3.all Shape: ",all.shape)
# sqft_above / bedroomss

all['sqft_bedroom'] = all['sqft_above']/all['bedrooms']
# sqft_above / bathrooms = fail

#all['sqft_bathroom'] = all['sqft_above']/all['bathrooms']
# waterfront area higher

waterfront_zipcode = all[(all['waterfront']!=0)]['zipcode'].unique()

bool_waterfront=[]

for zc in all['zipcode']:

  if zc in waterfront_zipcode:

    code = 1

  else : 

    code = 0

  bool_waterfront.append(code)

all['bool_wf_zc'] = bool_waterfront
# sqft_living / sqft_lot - fail

#all['sqft_living_lot'] = all['sqft_living']/all['sqft_lot']
#yr_recent - fail

#all['yr_recent'] = bigger_select(all,'yr_built','yr_renovated')
all = delete(all, 'id')##sqft_living15>

print("all shape: ",all.shape)

print("all shape: ",all.columns)
all = yearmonth_maker(all,'date')

all = delete(all, 'date')

print("all shape: ",all.shape)

print("all shape: ",all.columns)
all = logarize(all,'price','sqft_living','sqft_lot','sqft_above','sqft_lot15','sqft_living15','sqft_bedroom')

print("all shape: ",all.shape)

print("all shape: ",all.columns)
all = categorize(all, 'zipcode','yearmonth')

print("all shape: ",all.shape)

print("all shape: ",all.columns)
#all = cutagorize(all,24,'date','yearmonth')

print("all shape: ",all.shape)

print("all shape: ",all.columns)

all = standardize(all,'lat','long','yr_built')

print("all shape: ",all.shape)

print("all shape: ",all.columns)
all = binarize(all,"yr_renovated",'sqft_basement')

print("all shape: ",all.shape)

print("all shape: ",all.columns)
# bedroom per bathroom -> 

#all['bedroom_bathroom'] = all['bedrooms']/all['bathrooms']

# fail
#all = delete(all, 'lat', 'long')
## columns 확인

print(all.columns, all.shape)
## all을 train과 test로 다시 나누기

train = all[all['log_price'].isnull()==False]

test = all[all['log_price'].isnull()==True]



## 다시 dataset을 x_train, y_train, x_test로 나누기

x_train = train.drop(['log_price'],axis=1)

y_train = train['log_price']

x_test = test.drop(['log_price'],axis=1)



print(x_train.shape)

print(y_train.shape)

print(x_test.shape)
# XGB Parameter

xgb_params = {

    'eta': 0.02,

    'max_depth': 6,

    'subsample': 0.8,

    'colsample_bytree': 0.8,

    'objective': 'reg:linear',     # 회귀

    'eval_metirc': 'rmse',         # kaglle에서 요구하는 검증모델

    'silent': True                 # 학습 동안 메세지 출력 여부

}
# DMatrix

dtrain = xgb.DMatrix(x_train,y_train)

dtest = xgb.DMatrix(x_test)
# Feval(RMSE_exp)

def rmse_exp(predictions, dmat):

    labels = dmat.get_label()

    diffs = np.exp(predictions) - np.exp(labels)

    squared_diffs = np.square(diffs)

    avg = np.mean(squared_diffs)

    return ('rmse_exp', np.sqrt(avg))
# cv_output ## -> 내 모델의 성능을 객관적으로 파악하기 위함이다. -> test-rmse가 홈페이지에 있는 값과 비슷한 것.

cv_output = xgb.cv(xgb_params,

                   dtrain,                        

                   num_boost_round=5000,         # 학습 횟수

                   early_stopping_rounds=100,    # overfitting 방지

                   nfold=5,                      # 높을 수록 실제 검증값에 가까워지고 낮을 수록 빠름

                   verbose_eval=100,             # 몇 번째마다 메세지를 출력할 것인지

                   feval=rmse_exp,               # price 속성을 log scaling 했기 때문에, 다시 exponential

                   maximize=False,

                   show_stdv=False,              # 학습 동안 std(표준편차) 출력할지 말지

                   )

# scoring

best_rounds = cv_output.index.size

score = round(cv_output.iloc[-1]['test-rmse_exp-mean'], 2)



print(f'\nBest Rounds: {best_rounds}')

print(f'Best Score: {score}')
model = xgb.train(xgb_params, dtrain, num_boost_round=best_rounds)

y_pred = model.predict(dtest)

y_pred = np.exp(y_pred)

y_pred
# Read the Sample submission file

sample_submission = pd.read_csv('../input/sample_submission.csv')
sample_submission.head()
# Change the data

submission = pd.DataFrame(data={'id':sample_submission['id'],'price':y_pred})
submission.head()
# Save the data

file_name = 'submission.csv'

submission.to_csv(file_name,index=False)