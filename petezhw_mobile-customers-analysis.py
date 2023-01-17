# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
taobaoData = pd.read_csv("/kaggle/input/tianchi_mobile_recommend_train_user.csv")

taobaoData.head(5)
taobaoData.shape
taobaoData.isnull().sum()['user_geohash']/taobaoData.shape[0]
taobaoData.drop('user_geohash',axis=1,inplace=True)

taobaoData.head()
taobaoData.dtypes
taobaoData['user_id'] = taobaoData['user_id'].astype(str)

taobaoData['item_id'] = taobaoData['item_id'].astype(str)

taobaoData['behavior_type'] = taobaoData['behavior_type'].astype(int)

taobaoData['item_category'] = taobaoData['item_category'].astype(str)
taobaoData.isnull().sum()
taobaoData.drop_duplicates(['user_id','item_id','behavior_type','item_category','time'], keep='first')

taobaoData.count()
taobaoData['behavior_type'].unique()
type_arr = {1:'pv',2:'fav',3:'cart',4:'buy'}

taobaoData['behavior_type'] = taobaoData['behavior_type'].map(lambda x: type_arr[x])

taobaoData.head()
taobaoData['date'] = taobaoData['time'].map(lambda x: x.split(" ")[0])

taobaoData['hour'] = taobaoData['time'].map(lambda x: x.split(" ")[1])



taobaoData.head()
df_pv = taobaoData[taobaoData['behavior_type'] == 'pv']

pv = df_pv['user_id'].count()

print(pv)
uv = len(df_pv.groupby('user_id')['user_id'])

print(uv)
day_pv = df_pv.groupby('date').count()['user_id']
day_uv = df_pv.groupby(['date','user_id']).count().groupby('date').count()['item_id']
pv_uv = pd.concat([day_pv, day_uv], axis=1)

pv_uv = pv_uv.rename(columns={'user_id':'pv','item_id':'uv'})

pv_uv.head()
plt.rcParams["figure.figsize"]=10,6

ax=pv_uv.plot(kind='line',y='pv')

pv_uv.plot(kind='line',y='uv',color='red',secondary_y=True, ax=ax)
stat = taobaoData.groupby('behavior_type')['user_id'].count()

pv = stat['pv']

fav = stat['fav'] + stat['cart']

buy = stat['buy']

print (pv,fav,buy)

r1 = fav * 100/ pv

r2 = buy * 100 / fav

print (r1,r2)
# daily rate

stat_day = taobaoData.groupby(['date','behavior_type'])

stat_day1 = {}

for (k1,k2), group in stat_day:

    if not k1 in stat_day1:

        stat_day1[k1] = [k1,0,0,0]

    count = (group['user_id'].count())

    if k2 == 'buy':

        stat_day1[k1][1] += count

    if k2 == 'fav' or k2 == 'cart':

        stat_day1[k1][2] += count

    if k2 == 'pv':

        stat_day1[k1][3] += count
stat_day2 = []

for k,v in stat_day1.items():

    stat_day2.append( v )
stat_day3 = pd.DataFrame(stat_day2)

stat_day3['r1'] = stat_day3[2]*100/stat_day3[3]

stat_day3['r2'] = stat_day3[1]*100/stat_day3[2]

stat_day3.rename(columns={0:'date'}, inplace=True)

# stat_day3.head()

stat_day3[['date','r1','r2']].head()
#calculate twice purchase 

user_data = taobaoData[taobaoData['behavior_type'] == 'buy'].groupby('user_id').count()['item_id']

print(user_data.head())



total = user_data.count()

total2 = 0

for i in user_data:

    if i>= 2:

        total2 += 1

print(total2/total)

print(total)
# RF model

# F factor

# 1-5 purchases f1

# 5+ purchases f2

# R factor

# active in current 7 days r1

# active before current 7 days r2
# check the latest and oldest date

print(taobaoData['date'].max())

print(taobaoData['date'].min())
def splitDay(day):

    l = day.split("-")

    return [int(l[0]),int(l[1]),int(l[2])]
import datetime

d = splitDay(taobaoData['date'].max())

dmax = datetime.datetime(d[0],d[1],d[2])
def r(day):

    d = splitDay(day)

    da =  datetime.datetime(d[0],d[1],d[2])

    days = (dmax-da).days

    if days >= 7:

        return 2

    return 1
l = []

user_data = taobaoData[taobaoData['behavior_type'] == 'buy'].copy()

for day in user_data['date']:

    l.append(r(day))

    

user_data['r'] = l

user_data.head()
user_r = {}

for k,group in user_data.groupby('user_id')['r']:

    r = group.min()

    user_r[k] = r

    

dict(list(user_r.items())[0:5])
user_f = {}

for k, group in user_data.groupby('user_id')['item_id']:

    count = group.count()

    if count >=6 :

        f=2

    else:

        f=1

    user_f[k] = f

dict(list(user_f.items())[0:5])
user_rf = []

for k,v in user_f.items():

    user_rf.append( [k, user_r[k], user_f[k]] )

user_rf = pd.DataFrame(user_rf)

user_rf.rename(columns={0:'user_id',1:'r',2:'f'}, inplace=True)

user_rf.groupby(['r','f']).count()