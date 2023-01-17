import numpy as np

import pandas as pd
df_train = pd.read_csv('../input/sales_train.csv')
df_test = pd.read_csv('../input/test.csv')
df_train.head()
df_train.tail()
df_train.shape
df_train.shop_id.value_counts().shape
from datetime import datetime

 

year = lambda x: datetime.strptime(x, "%d.%m.%Y" ).year

day_of_week = lambda x: datetime.strptime(x, "%d.%m.%Y").weekday()

month = lambda x: datetime.strptime(x, "%d.%m.%Y" ).month

week_number = lambda x: datetime.strptime(x, "%d.%m.%Y").strftime('%V')



df_train['year'] = df_train['date'].map(year)

df_train['month'] = df_train['date'].map(month)

df_train['week_number'] = df_train['date'].map(week_number)

df_train['day_of_week'] = df_train['date'].map(day_of_week)



df_train.head()
df_train.item_id.value_counts()
df_train.date_block_num.value_counts().shape
df_test.head()
df_test.tail()
df_test.shape
df_test.shop_id.value_counts().shape
df_test.item_id.value_counts()
df_train.info()
df_train.describe()
df_train.shop_id.value_counts().shape
## Aggregate montly data for each item_id for each shop.

sum_df = df_train.groupby(['year','month', 'date_block_num','shop_id','item_id'], as_index=False).agg({'item_cnt_day': 'sum'})
sum_df.shape
df_train.shape
sum_df
import matplotlib.pyplot as plt
plt.boxplot(sum_df.item_cnt_day)
plt.plot(sum_df.item_cnt_day)
sum_df.columns
from sklearn.model_selection import train_test_split



#X= sum_df[['year', 'month', 'date_block_num', 'shop_id', 'item_id']]

X= sum_df[['date_block_num', 'shop_id', 'item_id']]



y = sum_df['item_cnt_day']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1) 
from sklearn.ensemble import RandomForestRegressor



clf=RandomForestRegressor()

clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)
clf.score(X_train, y_train)
clf.score(X_test, y_test)
from sklearn import metrics

print("Accuracy:",metrics.mean_squared_error(y_test, y_pred))
# http://www.statsmodels.org/devel/tools.html#measure-for-fit-performance-eval-measures

from statsmodels.tools import eval_measures
eval_measures.mse(y_test, y_pred)
eval_measures.rmse(y_test, y_pred)