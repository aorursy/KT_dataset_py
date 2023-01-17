# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

from pathlib import Path
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
def preprocess(df):
    df.date = pd.to_datetime(df.date)
    df['year'] = df.date.dt.year
    df['month'] = df.date.dt.month
    df['year_month'] = df.date.apply(lambda x: x.strftime('%Y-%m'))
    df['is_weekend'] = df.date.dt.day_name().isin(('Saturday', 'Sunday'))
    df['day_of_week'] = df.date.dt.dayofweek
    df['month_end'] = df.date.dt.is_month_end
    df['month_start'] = df.date.dt.is_month_start
    df['item'] = df.item.astype('category')
    df['store'] = df.store.astype('category')
    df['week_of_year'] = df.date.dt.weekofyear
    return df
PATH = Path("../input")
train_df = pd.read_csv(PATH / 'train.csv')
train_df.head()
train_df = preprocess(train_df)
train_df.head()

train_df.year_month.tail()
plt.figure(figsize=(8,5))
plt.hist(train_df.date, bins=12 * 5)
plt.show()
sales_per_date = train_df.groupby('date').sales.sum().reset_index()
sales_per_date.head()
sales_per_date = sales_per_date.sort_values('date')
sales_per_date['date_f'] = pd.factorize(sales_per_date.date)[0] + 1
plt.figure(figsize=(10,8))
ax = sns.regplot(x='date_f', y='sales', data=sales_per_date)
mapping = dict(zip(sales_per_date['date_f'], sales_per_date['date'].dt.date))
ax.set_xticklabels(pd.Series(ax.get_xticks()).map(mapping).fillna(''))
plt.show()
sales_per_store_over_time = train_df.groupby(['store', 'year_month']).sales.sum().reset_index()
total_sales = train_df.groupby('store').sales.sum()
highest_store = total_sales.reset_index().sort_values('sales', ascending=False).iloc[0,].store
lowest_store = total_sales.reset_index().sort_values('sales', ascending=True).iloc[0].store
plt.figure(figsize=(15,8))
for i in sales_per_store_over_time.store.unique():
    
    alpha = 0.5
    linewidth = 1
    color = 'grey'

    if i == lowest_store:
        color = 'red'
        alpha = 1
        linewidth = 2
    if i == highest_store:
        alpha = 1
        color = 'aqua'
        linewidth = 2

    store_values = sales_per_store_over_time[sales_per_store_over_time.store == i]

    plt.plot(store_values['year_month'], store_values['sales'], linewidth=linewidth, alpha=1, label=i, color=color)

plt.legend(loc='upper left')
plt.xticks(rotation=70)
plt.show()
list_of_sales_per_store = []
for i in sales_per_store_over_time.store.unique():
    store_values = sales_per_store_over_time[sales_per_store_over_time.store == i]
    list_of_sales_per_store.append(store_values.sales)
plt.figure(figsize=(15,10))
plt.stackplot(sales_per_store_over_time.date.unique(), list_of_sales_per_store, labels=sales_per_store_over_time.store.unique())
plt.legend(loc='upper left')
plt.show()
def smape(actual, target):
    return 100 * np.mean(2 * np.abs(actual - target)/(np.abs(actual) + np.abs(target)))
to_keep = ['store', 'item','year', 'month', 'is_weekend', 'day_of_week', 'month_end', 'month_start', 'week_of_year']
model = lightgbm.LGBMRegressor(n_jobs=-1, n_estimators=500, max_depth=8, objective='regression_l1', random_state=420)
valid_df = train_df[(train_df.year == 2017) & (train_df.month.isin((10, 11, 12)))]
train_df_dropped = train_df.drop(valid_df.index)
X = train_df_dropped[to_keep]
y = train_df_dropped['sales']
valid_X = valid_df[to_keep]
valid_y = valid_df['sales']
model.fit(X, y, eval_set=[(valid_X, valid_y)], eval_metric=['mape', smape], early_stopping_rounds=300)
feature_importances = pd.DataFrame({'importance': model.feature_importances_, 'name': X.columns})
sns.barplot(
    data=feature_importances.sort_values('importance', ascending=False), x='importance', y='name')
pred = model.predict(valid_X)
smape(valid_y, pred)
test_df = pd.read_csv(PATH / 'test.csv')
test_df = preprocess(test_df)
pred = model.predict(test_df[to_keep])
pd.DataFrame({'id': test_df['id'],'sales': pred}).to_csv('submission.csv', index=False)
