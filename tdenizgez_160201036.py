import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sales_train = pd.read_csv("../input/competitive-data-science-predict-future-sales/sales_train.csv")
items = pd.read_csv("../input/competitive-data-science-predict-future-sales/items.csv")

sales_train.head()
sales_train.tail()
items.head()
items=items.drop("item_name",axis=1)
items.head()
train_data = pd.merge(sales_train, items, on='item_id')
train_data.head()
train = train_data
#Outlier Değerleri Kaldırdık
train = train[train.item_price<100000]
train = train[train.item_cnt_day<1001]
median = train[(train.shop_id==32)&(train.item_id==2973)&(train.date_block_num==4)&(train.item_price>0)].item_price.median()
train.loc[train.item_price<0, 'item_price'] = median
import datetime
train_data.date = train_data.date.apply(lambda x:datetime.datetime.strptime(x, "%d.%m.%Y"))
train_data.head()
grouped = pd.DataFrame(train_data.groupby(['shop_id', 'date_block_num','item_id'])['item_cnt_day'].sum().reset_index())
total_item_cnt_mounth = grouped.groupby('date_block_num')['item_cnt_day'].sum()
#Total Shop Count:60
#Total Mounth Count:34
from math import ceil
fig, axes = plt.subplots(nrows=5, ncols=2, sharex=True, sharey=True, figsize=(16,20))
num_graph = 10
id_per_graph = ceil(grouped.shop_id.max() / num_graph)
count = 0
for i in range(5):
    for j in range(2):
        sns.pointplot(x='date_block_num', y='item_cnt_day', hue='shop_id', data=grouped[np.logical_and(count*id_per_graph <= grouped['shop_id'], grouped['shop_id'] < (count+1)*id_per_graph)], ax=axes[i][j])
        count += 1
print(total_item_cnt_mounth.head())
total_item_cnt_mounth_np = total_item_cnt_mounth.to_numpy()
total_item_cnt_mounth_np
mounths = np.arange(34)
mounths
plt.plot(mounths,total_item_cnt_mounth_np)
plt.xlabel('Mounth')
plt.ylabel('Total Sale Count')
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(mounths,total_item_cnt_mounth_np, test_size = 1/3, random_state = 123, shuffle=1)
X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
model = LinearRegression()
model.fit(X_train, y_train)
model.score(X_train,y_train)
model.predict([[34]])
pred = model.predict(X_test)
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': pred.flatten()})
df
from sklearn.metrics import mean_squared_error
from math import sqrt
mse = mean_squared_error(y_test, pred)
mse
rmse = sqrt(mean_squared_error(y_test, pred))
rmse
