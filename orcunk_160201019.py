import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from pandas import read_csv
import seaborn as sns


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Öncelikle veriyi okuyoruz
item_categories=pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')
items=pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')
sales_train=pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')
sample_submission=pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv')
shops=pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')
test=pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')
# item_cnt_day alanı ve  item_price alanı sıfırdan küçük olamaz bu sebeple veriyi yeniden düzenliyoruz
sales_train = sales_train[sales_train['item_cnt_day']>0]
sales_train = sales_train[sales_train['item_price']>0]

#Verilere baktığımda verilerin ilişkisel veri tabanları gibi tasarlanmış olduklarını gördüm bu sebeple veriyi birleştiriyoruz
train_full = pd.merge(sales_train, items, how='left', on=['item_id','item_id'])
train_full = pd.merge(train_full, item_categories, how='left', on=['item_category_id','item_category_id'])
train_full = pd.merge(train_full, shops, how='left', on=['shop_id','shop_id'])
train_full['total_price']=train_full['item_price']*train_full['item_cnt_day']

#Tarihin date verisi olarak algılanması için ayarları yapıyoruz
train_full['date'] = pd.to_datetime(train_full['date'], format='%d.%m.%Y')
train_full['month'] = train_full['date'].dt.month
train_full['year'] = train_full['date'].dt.year
train_full['day'] = train_full['date'].dt.day
train_full.tail()
train_full.head()
train_full.info()
#Verilerimizi tek bir dataframede toparladık
# Buradaki grafikle hangi aylarda en fazla satış yapıldığını görebilmekteyiz
plt.figure(figsize=(35,10))
sns.countplot(x='date_block_num', data=train_full);
plt.xlabel('Aylar')
plt.ylabel('Satışlar')
plt.title('Aylara göre Satışlar')
plt.show()
#Hangi mağazaların ne kadarlık satış yaptıklarını buradan görebilmekteyiz.
sales_total_price = pd.DataFrame(train_full.groupby(['shop_id'])['total_price'].sum().reset_index())
plt.figure(figsize=(35,10))
plt.xlabel('Mağaza ID')
plt.ylabel('Toplam Kazanç')
plt.title('Mağazaların Aylara göre Toplam Kazançları')
sns.barplot(x="shop_id", y="total_price", data=sales_total_price , order=sales_total_price['shop_id'])
plt.show()
#Hangi mağazaların kaçar adet satış yaptıklarını buradan görebilmekteyiz.
sales_total = pd.DataFrame(train_full.groupby(['shop_id'])['item_cnt_day'].sum().reset_index())
plt.figure(figsize=(35,10))
plt.xlabel('Mağaza ID')
plt.ylabel('Toplam Satış Sayısı')
plt.title('Mağazaların Aylara göre Toplam Satışları')
sns.barplot(x="shop_id", y="item_cnt_day", data=sales_total , order=sales_total_price['shop_id'])
plt.show()
train_full = train_full[['date_block_num','shop_id','item_id','total_price','item_category_id','item_cnt_day']]
train_full
from keras.models import Sequential
from keras.layers import Dense



X = train_full.iloc[:, :5]  
y = train_full.iloc[:, 5:6]  

model = Sequential()
model.add(Dense(32, input_dim=5, activation='relu'))   
model.add(Dense(16, activation='relu'))                
model.add(Dense(1, activation='sigmoid'))            

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])   

model.fit(X, y, epochs=5)  
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))
from keras.utils.vis_utils import plot_model

plot_model(model, to_file='pima_model_plot.png', 
           show_shapes=True, show_layer_names=True)

pima_model_plot=plt.imread("pima_model_plot.png")
plt.figure(figsize=(12,10))
plt.xticks([])
plt.yticks([])
plt.imshow(pima_model_plot)
plt.show()
