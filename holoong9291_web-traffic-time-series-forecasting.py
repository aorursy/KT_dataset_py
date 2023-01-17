# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from pylab import rcParams

import matplotlib.pyplot as plt

from scipy.stats import probplot

from fbprophet import Prophet



%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
os.system('unzip -d /kaggle/input /kaggle/input/web-traffic-time-series-forecasting/train_2.csv.zip')

os.system('unzip -d /kaggle/input /kaggle/input/web-traffic-time-series-forecasting/train_1.csv.zip')

os.system('unzip -d /kaggle/input /kaggle/input/web-traffic-time-series-forecasting/key_2.csv.zip')

os.system('unzip -d /kaggle/input /kaggle/input/web-traffic-time-series-forecasting/key_1.csv.zip')

os.system('unzip -d /kaggle/input /kaggle/input/web-traffic-time-series-forecasting/sample_submission_2.csv.zip')

os.system('unzip -d /kaggle/input /kaggle/input/web-traffic-time-series-forecasting/sample_submission_1.csv.zip')
folder = '../input/'



train2 = pd.read_csv(folder+'train_2.csv')
train2 = train2.join(train2['Page'].str.rsplit('_',n=3,expand=True)).rename(columns={0:'article',1:'source',2:'access',3:'agent'})

train2 = train2.drop('Page', axis=1)

train2.info()
# %%time



# rcParams['figure.figsize'] = 10, 5



# count = 0

# idxs = []

# for i in range(len(train2)):

#     _row = train2.iloc[i]

#     if sum(_row.isnull())>0:

#         count += 1

#         idxs.append(i)

# #         if count>5:

# #             break

# #print(len(idxs),idxs)



# _tmp = train2

# for ii,idx in enumerate(idxs):

#     print(ii,len(idxs))

#     features,values=[],[]

#     dss,ys,idxs_null = [],[],[]

#     # reshape row to dataframe

#     for col,value in train2.iloc[idx].items():

#         if col in ['Page','article','source','access','agent']:

#             features.append(col)

#             values.append(value)

#             continue

#         dss.append(col)

#         ys.append(value)

#         if np.isnan(value):

#             idxs_null.append(len(ys)-1)

#     _row = pd.DataFrame({'ds':dss,'y':ys})

#     #print(len(_row))

#     #print(idxs_null[-1])

#     #_first = idxs_null[-1]+1

#     #_row_train = _row[_first:]

#     #_row_test = _row[:_first]

#     #print(len(_row_train),len(_row_test))

#     m = Prophet()

#     m.fit(_row)

#     #forecast = m.predict(pd.concat([_row_test[['ds']],_row_train[['ds']]]))

#     forecast = m.predict(_row[['ds']])

#     #m.plot(forecast)

#     for i in idxs_null:

#         ys[i] = forecast[forecast.ds==_row.iloc[i].ds].iloc[0].yhat

#     _row = pd.Series(values+ys, index=features+dss)

#     #print(sum(_row.isnull()))

#     _tmp.iloc[idx] = _row

#     #break



# train2 = _tmp



# for i in idxs:

#     _row = _tmp.iloc[i]

#     if sum(_row.isnull())>0:

#         print(sum(_row.isnull()))
cols = list(set(train2.columns.tolist())-set(['article','source','access','agent']))

train2[cols] = train2[cols].fillna(method='bfill', axis=1)

train2[cols] = train2[cols].fillna(method='ffill', axis=1)
train2 = train2.fillna(0)
# for col in train2.columns:

#     if train2[col].dtype != 'object':

#         train2[col] = train2[col].clip(0)

train2[cols] = train2[cols].clip(0)
# 部分列做类型转换

train2[['article','source','access','agent']] = train2[['article','source','access','agent']].astype('category')

train2[cols] = train2[cols].apply(pd.to_numeric, downcast='unsigned')

train2.info()
visits = train2[cols].stack().reset_index(level=1)

visits.columns = ['date','visits']

#visits.date = visits.date.astype(np.datetime64)

visits.date = visits.date.astype('category')

visits.visits = visits.visits.astype(np.int32)

visits.info()
train2 = train2.drop(cols, axis=1).join(visits)

del visits

train2.info(memory_usage='deep')
# import seaborn as sns

# sns.set(style="darkgrid")

# sns.lineplot(x="date", y="visits", hue="article", 

#              data=train2[['article','date','visits']][(train2.article=='董子健')|(train2.article=='何廣沛')|(train2.article=='李宗伟')])
# # 2017-09-11, 2017-09-12

# for k,group in train2.groupby(['article','source','access','agent']):

# #for k,group in train2.groupby('article'):

#     _article,_source,_access,_agent,_visits = k[0],k[1],k[2],k[3],0

#     #_date1,_date2 = np.datetime64('2017-09-11'),np.datetime64('2017-09-12')

#     _date1 = np.datetime64('2017-11-13')

#     train2 = train2.append({'article':_article,'source':_source,'access':_access,'agent':_agent,'visits':_visits,'date':_date1}, ignore_index=True)

#     #train2 = train2.append({'article':_article,'source':_source,'access':_access,'agent':_agent,'visits':_visits,'date':_date2}, ignore_index=True)



# 2017-09-13 ~ 2017-11-13

key2 = pd.read_csv(folder+'key_2.csv')

key2 = key2.join(key2['Page'].str.rsplit('_',n=4,expand=True)).rename(columns={0:'article',1:'source',2:'access',3:'agent',4:'date'})

key2 = key2.drop(['Page','Id'], axis=1)

key2['visits'] = 0

key2.visits = key2.visits.astype(np.int8)

# 部分列做类型转换

key2[['date','article','source','access','agent']] = key2[['date','article','source','access','agent']].astype('category')

key2.info(memory_usage='deep')
import gc

gc.collect()
from pandas.api.types import union_categoricals



# matrix = pd.DataFrame({})

# matrix['article'] = pd.Series(union_categoricals([train2.article,key2.article]))

# matrix['source'] = pd.Series(union_categoricals([train2.source,key2.source]))

# matrix['access'] = pd.Series(union_categoricals([train2.access,key2.access]))

# matrix['agent'] = pd.Series(union_categoricals([train2.agent,key2.agent]))

# matrix['date'] = pd.concat([train2.date,key2.date])

# matrix['visits'] = pd.concat([train2.visits,key2.visits])

# del key2,train2

# matrix.info()



matrix = pd.DataFrame({})

print('article ....')

matrix['article'] = pd.Series(union_categoricals([train2.article,key2.article]))

del train2['article'],key2['article']

print('source ....')

matrix['source'] = pd.Series(union_categoricals([train2.source,key2.source]))

del train2['source'],key2['source']

print('access ....')

matrix['access'] = pd.Series(union_categoricals([train2.access,key2.access]))

del train2['access'],key2['access']

print('agent ....')

matrix['agent'] = pd.Series(union_categoricals([train2.agent,key2.agent]))

del train2['agent'],key2['agent']

print('date ....')

matrix['date'] = pd.Series(union_categoricals([train2.date,key2.date]))

del train2['date'],key2['date']

print('visits ....')

#matrix['visits'] = pd.concat([train2.visits,key2.visits])

matrix['visits'] = train2.visits.append(key2.visits, ignore_index=True)

del train2['visits'],key2['visits']

del train2,key2

print('show info ....')

matrix.info(memory_usage='deep')
print(len(matrix[matrix.article=='2NE1'].access.unique()))

print(len(matrix[matrix.article=='2NE1'].agent.unique()))

print(len(matrix[matrix.article=='2NE1'].source.unique()))

print(len(matrix[matrix.article=='2NE1'].date.unique()))

print(len(matrix[matrix.article=='2NE1'].visits.unique()))

print(len(matrix[matrix.article=='2NE1']))