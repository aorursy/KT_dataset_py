import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

sales = pd.read_csv("../input/competitive-data-science-predict-future-sales/sales_train.csv", parse_dates=['date'], infer_datetime_format=True,dayfirst=True) 
sales.head()
test = pd.read_csv("../input/competitive-data-science-predict-future-sales/test.csv")
test.head()
#Sales subset

sales_column = ['date_block_num','shop_id','item_id','item_price','item_cnt_day']
sales.drop_duplicates(sales_column, keep='first', inplace=True)
sales.reset_index(drop=True, inplace=True)
test_column = ['shop_id','item_id']
test.drop_duplicates(test_column, keep='first', inplace=True)
test.reset_index(drop=True, inplace=True)
sales.loc[sales.item_price < 0,'item_price'] = 0
sales['item_cnt_day'] = sales['item_cnt_day'].clip(0,1000)
sales['item_price'].max()
sales['item_price'] = sales['item_price'].clip(0,300000)
df = sales.groupby([sales.date.apply(lambda x: x.strftime('%Y-%m')), 'item_id', 'shop_id']).sum().reset_index()
df.columns
df = df[['date', 'item_id', 'shop_id', 'item_cnt_day']]
#Değerler - cnt_day , kolon - date , indexleri ise item/shop id

df = df.pivot_table(index=['item_id', 'shop_id'], columns='date', values='item_cnt_day', fill_value=0).reset_index()

df
data = pd.merge(test, df, on=['item_id', 'shop_id'], how='left').fillna(0)
#Bağımlı değişken

y = y = data.iloc[:,-1:]

#Bağımsız değişken

x = data.iloc[:,3:]

x.drop(['2015-10'], axis = 1, inplace = True)
y
x
#Decision Tree Regression için hazırlık

x = x.values

y = y.values.reshape(-1,1)
from sklearn.tree import DecisionTreeRegressor



tree = DecisionTreeRegressor()



tree.fit(x,y)



y_head = tree.predict(x)



y_head = y_head.reshape(-1,1)
#Root Mean Square Error

def rmse(predict,real):

    return np.sqrt(np.mean((predict-real) ** 2))



rmse(y_head,y)
d = data.loc[:,["ID"]]

d["item_cnt_month"] = y_head
submission = pd.DataFrame(data=d)

submission.to_csv('submission.csv', index=False)

submission.head()