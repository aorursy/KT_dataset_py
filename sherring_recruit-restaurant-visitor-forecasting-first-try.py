# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import *

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

air_visit=pd.read_csv('../input/air_visit_data.csv')

air_reserve=pd.read_csv('../input/air_reserve.csv')

air_store=pd.read_csv('../input/air_store_info.csv')

date=pd.read_csv('../input/date_info.csv')

store=pd.read_csv('../input/store_id_relation.csv')

hpg_reserve=pd.read_csv('../input/hpg_reserve.csv')

hpg_store=pd.read_csv('../input/hpg_store_info.csv')

submission=pd.read_csv('../input/sample_submission.csv')

air_visit.head()

air_reserve.head()

air_store.head()





hpg_reserve.head()

hpg_store.head()

date.head()

submission.head()

air_store.head()





# Any results you write to the current directory are saved as output.
# cacluate reserve 

air_reserve['visit_datetime']=pd.to_datetime(air_reserve['visit_datetime'])

air_reserve['reserve_datetime']=pd.to_datetime(air_reserve['reserve_datetime'])

air_reserve['visit_date']=air_reserve['visit_datetime'].dt.date

air_reserve['reserve_visit_diff_days']=air_reserve.apply(lambda r: (r['visit_datetime'] - r['reserve_datetime']).days, axis=1)

air_reserve.head()

air_reserve_gp1=air_reserve.groupby(['air_store_id','visit_date'],as_index=False)['reserve_visitors'].sum()

air_reserve_gp2=air_reserve.groupby(['air_store_id','visit_date'],as_index=False)['reserve_visit_diff_days'].mean()

air_reserve_gp=pd.merge(air_reserve_gp1,air_reserve_gp2,how='inner',on=['air_store_id','visit_date'])

air_reserve_gp.head()
#merge visitor and reserve, can see some dates no reserve 

air_visit['visit_date']=pd.to_datetime(air_visit['visit_date'])

air_visit['visit_week']=air_visit['visit_date'].dt.dayofweek

air_visit['visit_month']=air_visit['visit_date'].dt.month

air_visit['visit_year']=air_visit['visit_date'].dt.year

air_visit['visit_date']=air_visit['visit_date'].dt.date

air_visit_reserve=pd.merge(air_visit,air_reserve_gp,how='left',on=['air_store_id','visit_date'])

air_visit_reserve.head()


# transform air_store Converting categorical feature to numeric

# gernre=air_store.air_genre_name.unique()

# print(gernre)

# gernre_mapping = {"Italian/French": 1, "Dining bar": 2, "Yakiniku/Korean food": 3, "Cafe/Sweets": 4, "Izakaya": 5,\

#                   "Okonomiyaki/Monja/Teppanyaki":6,"Bar/Cocktail":7,"Japanese food":8,"Creative cuisine":9,"Western food":10,\

#                  "International cuisine":11,"Asian":12,"Karaoke/Party":13,"Other":14}

# air_store['air_genre_name'] = air_store['air_genre_name'].map(gernre_mapping)

# air_store['air_genre_name'] = air_store['air_genre_name'].fillna(0)



# air_store.head()

# gernre=air_store.air_area_name.unique()

# print(gernre)

# convert feature as numerical 

air_store=pd.read_csv('../input/air_store_info.csv')

lbl = preprocessing.LabelEncoder()

air_store['air_genre_name'] = lbl.fit_transform(air_store['air_genre_name'])

air_store['air_area_name'] = lbl.fit_transform(air_store['air_area_name'])

date=pd.read_csv('../input/date_info.csv')

# dow=date.day_of_week.unique()

# print(dow)

day_mapping={"Monday":1,"Tuesday":2,"Wednesday":3,"Thursday":4,"Friday":5,"Saturday":6,"Sunday":0}

date['dow']=date['day_of_week'].map(day_mapping)

date['calendar_date']=pd.to_datetime(date['calendar_date']).dt.date

date.head()
#merge air store info 

air_visit_reserve_store=pd.merge(air_visit_reserve,air_store,how='left',on=['air_store_id'])

air_visit_reserve_store.head()

#merge with holiday

air_visit_reserve_store=pd.merge(air_visit_reserve_store,date,left_on='visit_date',right_on='calendar_date')

air_visit_reserve_store.head()

#caculate averge reserve vistitors and diff days per dow

air_visit_reserve_store_gp=air_visit_reserve_store.groupby(['air_store_id','dow'],as_index=False)['reserve_visitors','reserve_visit_diff_days'].mean().rename(columns={'reserve_visitors':'mean_reserve_visitors','reserve_visit_diff_days':'mean_reserve_visit_diff_days'})

air_visit_reserve_store_gp.head()

air_visit_reserve_store=pd.merge(air_visit_reserve_store,air_visit_reserve_store_gp,on=['air_store_id','dow'],how='left')

air_visit_reserve_store.head()



air_visit_reserve_store['reserve_visitors'] = air_visit_reserve_store['reserve_visitors'].fillna(air_visit_reserve_store['mean_reserve_visitors'])

air_visit_reserve_store['reserve_visit_diff_days'] = air_visit_reserve_store['reserve_visit_diff_days'].fillna(air_visit_reserve_store['mean_reserve_visit_diff_days'])

air_visit_reserve_store.head()
     

air_visit_reserve_store['reserve_visitors'] = air_visit_reserve_store['reserve_visitors'].fillna(0)

air_visit_reserve_store['reserve_visit_diff_days']=air_visit_reserve_store['reserve_visit_diff_days'].fillna(0) 

train=air_visit_reserve_store

lbl = preprocessing.LabelEncoder()

train['air_store_id2']=lbl.fit_transform(train['air_store_id'])

train['date_int'] = train['visit_date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)

X_train = train.drop(['calendar_date', 'day_of_week','visitors','visit_date','air_store_id','mean_reserve_visitors','mean_reserve_visit_diff_days'], axis=1)# build test: -1 means last index

Y_train = train['visitors'].values

X_train.head()



train_columns=X_train.columns

print(train_columns)
# build test: -1 means last index

submission['air_store_id']=submission.id.map(lambda x: '_'.join(x.split('_')[:-1]))

submission['visit_date']=submission.id.map(lambda x: x.split('_')[-1])

submission['visit_date']=pd.to_datetime(submission['visit_date'])

submission['visit_week']=submission['visit_date'].dt.dayofweek

submission['visit_month']=submission['visit_date'].dt.month

submission['visit_year']=submission['visit_date'].dt.year

submission['visit_date']=submission['visit_date'].dt.date



test=pd.merge(submission,date,how='left',left_on='visit_date',right_on='calendar_date')

test=pd.merge(test,air_store,how='left',on=['air_store_id'])

test.head()

test=pd.merge(test,air_reserve_gp,how='left',on=['air_store_id','visit_date'])

#print(test.groupby('reserve_visitors').count())

test.head()

test_gp=pd.merge(test,air_visit_reserve_store_gp,on=['air_store_id','dow'],how='left')

test_gp.head()

test_gp['reserve_visitors'] = test_gp['reserve_visitors'].fillna(air_visit_reserve_store['mean_reserve_visitors'])

test_gp['reserve_visit_diff_days'] = test_gp['reserve_visit_diff_days'].fillna(air_visit_reserve_store['mean_reserve_visit_diff_days'])

test_gp['reserve_visitors'] = test_gp['reserve_visitors'].fillna(0)

test_gp['reserve_visit_diff_days']=test_gp['reserve_visit_diff_days'].fillna(0) 



lbl = preprocessing.LabelEncoder()

test_gp['air_store_id2']=lbl.fit_transform(test_gp['air_store_id'])

test_gp['date_int'] = test_gp['visit_date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)



# X_test=test.drop(['id','calendar_date','day_of_week','visitors','air_store_id','visit_date'],axis=1)

# Y_test=test['visitors']

X_test=test_gp[train_columns]

X_test.head()
import xgboost as xgb

#print(X_train.count())

split = 200000

x_train, y_train, x_valid, y_valid = X_train[:split], Y_train[:split], X_train[split:], Y_train[split:]

d_train = xgb.DMatrix(x_train, label=y_train)

d_valid = xgb.DMatrix(x_valid, label=y_valid)

d_test = xgb.DMatrix(X_test)

params = {}

params['eta'] = 0.02

params['objective'] = 'reg:linear'

#params['eval_metric'] = 'mae' #mean absolute error

params['max_depth'] = 4

params['silent'] = 1



watchlist = [(d_train, 'train'), (d_valid, 'valid')]

clf = xgb.train(params, d_train, 2000, watchlist, early_stopping_rounds=100, verbose_eval=10)

p_test = clf.predict(d_test)

print(p_test)
output=pd.read_csv('../input/sample_submission.csv')

output['visitors']=p_test

output.to_csv('submission.csv',index=False,float_format='%.4f')

output.head()
# EDA

#Visualization libs

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib import gridspec

%matplotlib inline

from datetime import datetime

import pandas as pd 

import numpy as np

#Visitor each day

air_visit=pd.read_csv('../input/air_visit_data.csv')

air_visit_day= air_visit.groupby(['visit_date'], as_index=False).agg({'visitors': np.sum})

plt.figure(figsize=(12,6))

plt.plot(pd.to_datetime(air_visit_day['visit_date']).dt.date,air_visit_day['visitors'])

plt.gcf().autofmt_xdate()

plt.xlabel('visit_date', fontsize=12)

plt.ylabel("Sum of Visitors")

plt.title("Visitor each day")

# print(type(air_visit))

# print(type(air_visit_day))
# vivist per dow The day of the week with Monday=0, Sunday=6, Statuday and friday and sunday are hot day

import seaborn as sns

air_visit['dow']=pd.to_datetime(air_visit['visit_date']).dt.dayofweek

air_visit_dow=air_visit.groupby(['dow'],as_index=False).agg({'visitors':np.sum})

#print(air_visit_dow['dow'])

dow_labels = ['Monday', 'Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

#plt.plot(air_visit_dow['dow'],air_visit_dow['visitors'])

#plt.barplot(air_visit_dow['dow'],air_visit_dow['visitors'])

plt.figure(figsize=(12,6))

sns.barplot(air_visit_dow['dow'],air_visit_dow['visitors'])

plt.xticks(air_visit_dow['dow'],dow_labels,rotation=45)

plt.xlabel('dow', fontsize=12)

plt.ylabel("Sum of Visitors")

plt.title("Day of Week")



# May and June are less vistiors then other month

air_visit['month']=pd.to_datetime(air_visit['visit_date']).dt.month

air_visit_month=air_visit.groupby(['month'],as_index=False).agg({'visitors':np.sum})

month_labels = ['January','February','March','April','May','June','July','August','September','October','November','December']

air_visit_month.set_index(air_visit_month['month'])

# print(air_visit_month['month'])

# print(air_visit_month['month'].index)

plt.figure(figsize=(12,6))

sns.barplot(air_visit_month['month'],air_visit_month['visitors'])

plt.xticks(air_visit_month['month']-1,month_labels,rotation=45) # month start from 1, but label need map from 0

plt.xlabel('month', fontsize=12)

plt.ylabel("Sum of Visitors")

plt.title("Monthly visitors")

print(air_visit_month[air_visit_month['month']==8])

print(air_visit_month[air_visit_month['month']==7])
# check if test data set's dates already have reseration

##32019 recrods need predict, but only 1793 of them (5%) have reservtation. so concerned if need this featur:visitor_diff to reference

submission=pd.read_csv('../input/sample_submission.csv')

submission['air_store_id']=submission.id.map(lambda x: '_'.join(x.split('_')[:-1]))

submission['visit_date']=submission.id.map(lambda x: x.split('_')[-1])

submission['visit_date']=pd.to_datetime(submission['visit_date'])

submission['visit_week']=submission['visit_date'].dt.dayofweek

submission['visit_month']=submission['visit_date'].dt.month

submission['visit_year']=submission['visit_date'].dt.year

submission['visit_date']=submission['visit_date'].dt.date

submission.head()

air_reserve=pd.read_csv('../input/air_reserve.csv')

air_reserve['visit_datetime']=pd.to_datetime(air_reserve['visit_datetime'])

air_reserve['reserve_datetime']=pd.to_datetime(air_reserve['reserve_datetime'])

air_reserve['visit_date']=air_reserve['visit_datetime'].dt.date

air_reserve['reserve_visit_diff_days']=air_reserve.apply(lambda r: (r['visit_datetime'] - r['reserve_datetime']).days, axis=1)

air_reserve.head()

sub_reserve=pd.merge(submission,air_reserve,how='inner',on=['air_store_id','visit_date'])

sub_reserve.head()

print(sub_reserve.count()) # 1793 records have reserve

print(submission.count()) #32019 recrods need predict, so only 1793 of them (5%)will have reserve_visitor_diff to reference
# check air store count with genre and area

air_store=pd.read_csv('../input/air_store_info.csv')

air_store.head()

air_store_area=air_store['air_area_name'].value_counts().reset_index().sort_index()

air_store_area.columns=['air_area_name','store_counts']

air_store_genre=air_store['air_genre_name'].value_counts().reset_index().sort_index()

air_store_genre.columns=['air_genre_name','store_counts']

air_store_genre.head()



fig,ax = plt.subplots(1,2)

sns.barplot(air_store_area['store_counts'],air_store_area['air_area_name'][:15] ,ax=ax[0])

sns.barplot(air_store_genre['store_counts'],air_store_genre['air_genre_name'] ,ax=ax[1])

#fig.set_size_inches(w, h) 

fig.set_size_inches(15,10,forward=True)

# ax[0].set_ylabel('Number of Restaurent')

# ax[1].set_ylabel('Number of Restaurent')

# let's check visitor count with different area and genre

air_visit.head()

air_store.head()

air_visit_restaurant=pd.merge(air_visit,air_store,how='left',on=['air_store_id'])

air_visit_restaurant.head()

#as_index=False is effectively “SQL-style” grouped output, default sort by group key 

air_visit_restaurant_genre=air_visit_restaurant.groupby(['air_genre_name'],as_index=False).agg({'visitors':np.sum})

air_visit_restaurant_area=air_visit_restaurant.groupby(['air_area_name'],as_index=False).agg({'visitors':np.sum})

air_visit_restaurant_genre=air_visit_restaurant_genre.sort_values(['visitors'],ascending=False)

air_visit_restaurant_area=air_visit_restaurant_area.sort_values(['visitors'],ascending=False)

#print(air_visit_restaurant_genre.dtypes)

#print(air_visit_restaurant_genre)

fig,ax = plt.subplots(1,2)

sns.barplot(air_visit_restaurant_genre['visitors'],air_visit_restaurant_genre['air_genre_name'][:15] ,ax=ax[0])

sns.barplot(air_visit_restaurant_area['visitors'],air_visit_restaurant_area['air_area_name'][:15] ,ax=ax[1])

#fig.set_size_inches(w, h) 

fig.set_size_inches(15,10,forward=True)
# check store count/visitor conunt with genre: more store count , more vistior count 

air_store_genre

#sns.barplot(x="sex", y="survived", hue="class", data=titanic);
air_visit_restaurant_genre
# check relationship with genre with area: so hard to see, need refrator later.

air_visit_restaurant_genre

air_visit_restaurant_area

# air_genre_area=pd.merge(air_visit_restaurant_genre,air_visit_restaurant_area,how='outer',on=['visitors'])

# air_genre_area

air_visit_restaurant

result = air_visit_restaurant.pivot_table(index='air_genre_name',columns='air_area_name',values='visitors')

result

sns.heatmap(result, annot=True, fmt="g", cmap='viridis')

plt.show()
air_visit_restaurant