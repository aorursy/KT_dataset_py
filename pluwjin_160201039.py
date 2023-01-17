import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
train_data = pd.read_csv("../input/competitive-data-science-predict-future-sales/sales_train.csv")
test_data = pd.read_csv("../input/competitive-data-science-predict-future-sales/test.csv")
train_data.groupby(['date_block_num']).sum().reset_index()
train_data[(train_data['shop_id']==25)]
# Satış verilerinin aylık hale geitirilmesi
train_monthly = train_data.groupby(['date_block_num','item_id','shop_id']).sum().reset_index()[['shop_id','item_id','date_block_num','item_cnt_day']]
train_monthly = train_monthly.sort_values(['shop_id','item_id','date_block_num'])
train_monthly = train_monthly.rename(columns={'item_cnt_day' : 'item_cnt_mntly', 'date_block_num': 'month_no'})
print(train_monthly)
sample=train_monthly[train_monthly['item_id']==30]
print(sample[sample['shop_id']==59])

train_x=train_monthly.pivot_table(index=['shop_id','item_id'],values=['item_cnt_mntly'],columns=['month_no'],aggfunc='sum')
train_x.reset_index(inplace=True)
train_x=pd.merge(test_data,train_x,on=['item_id','shop_id'], how='left').drop(['shop_id','item_id','ID'],axis=1)
train_x=train_x.fillna(0)
print(train_x.shape)
train_numpy=np.array(train_x)
print(train_numpy)
train_x

# Kasım Ayı tahmini için önceki yılların kasım ayı değerlerinin ortalaması label olarak alınır.
labels=np.zeros((len(train_numpy),1))
for row in range(0,len(train_numpy)):
   labels[row][0]=(train_numpy[row][10]+train_numpy[row][22])/2.0

train_d = np.expand_dims(train_x.values[:,1:],axis = 2)

print(train_d.shape)
print(labels.shape)

import tensorflow as tf

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv1D(32,3,activation='relu',input_shape=(34,1)))
model.add(tf.keras.layers.LSTM(units=32))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(1))

model.compile(loss = 'mse',optimizer = 'adam', metrics = ['mean_squared_error','cosine_similarity','mean_absolute_error'])
model.summary()
history=model.fit(train_d,labels,batch_size=2048,epochs=20)
import matplotlib.pyplot as plt
plt.plot(history.history['mean_squared_error'])
plt.title('MSE/EPOCH')
plt.ylabel('MSE')
plt.xlabel('epoch')
plt.show()