import seaborn as sea
import numpy as num
import pandas as pan 
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.models import Sequential
from keras.layers import Dense, Dropout
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

sales_train = pan.read_csv("../input/competitive-data-science-predict-future-sales/sales_train.csv")
items = pan.read_csv("../input/competitive-data-science-predict-future-sales/items.csv")
sample_submission = pan.read_csv("../input/competitive-data-science-predict-future-sales/sample_submission.csv")
test = pan.read_csv("../input/competitive-data-science-predict-future-sales/test.csv")
item_catalog = pan.read_csv("../input/competitive-data-science-predict-future-sales/item_categories.csv")
shops = pan.read_csv("../input/competitive-data-science-predict-future-sales/shops.csv")                                    
sales_train.head()
sales_train.tail()
sales_train = sales_train[sales_train['item_cnt_day']>0]
sales_train = sales_train[sales_train['item_cnt_day']<1000]
sales_train = sales_train[sales_train['item_price']>0]
sales_train = sales_train[sales_train['item_price']<100000]
egitim_verisi = pan.merge(sales_train, items, how='left', on=['item_id'])
egitim_verisi = pan.merge(egitim_verisi, item_catalog, how='left', on=['item_category_id'])
egitim_verisi = pan.merge(egitim_verisi, shops, how='left', on=['shop_id'])
egitim_verisi['date'] = pan.to_datetime(egitim_verisi['date'], format='%d.%m.%Y')
egitim_verisi.head()
egitim_verisi=egitim_verisi.drop("item_name",axis=1)
egitim_verisi=egitim_verisi.drop("item_category_name",axis=1)
egitim_verisi=egitim_verisi.drop("shop_name",axis=1)

egitim_verisi.head()
birlestirilmis = pan.DataFrame(egitim_verisi.groupby(['shop_id', 'date_block_num','item_id'])['item_cnt_day'].sum().reset_index())
aylık_toplam_satis = birlestirilmis.groupby('date_block_num')['item_cnt_day'].sum()
toplam_ay_sayisi = num.arange(34)
aylık_toplam_satis = aylık_toplam_satis.to_numpy()
aylık_toplam_satis
x_egitimVerisi, x_testVerisi, y_egitimVerisi, y_testVerisi = train_test_split(toplam_ay_sayisi,aylık_toplam_satis, test_size = 25/100, random_state = 123, shuffle=1)
print(x_egitimVerisi.shape,x_testVerisi.shape,y_egitimVerisi.shape,y_testVerisi.shape)
x_egitimVerisi = x_egitimVerisi.reshape(-1, 1);x_testVerisi = x_testVerisi.reshape(-1, 1)
y_egitimVerisi = y_egitimVerisi.reshape(-1, 1);y_testVerisi = y_testVerisi.reshape(-1, 1)
print(x_egitimVerisi.shape,x_testVerisi.shape,y_egitimVerisi.shape,y_testVerisi.shape)
LR_Modeli = LinearRegression()
LR_Modeli.fit(x_egitimVerisi, y_egitimVerisi)
print("Skor: ", LR_Modeli.score(x_egitimVerisi,y_egitimVerisi))
tahmin=LR_Modeli.predict(x_testVerisi)
hata_orani = mean_squared_error(y_testVerisi, tahmin)
print("Hata Değeri: ",hata_orani)


print("Bir Sonraki Ay Tahmin Edilen Toplam Satış: " , LR_Modeli.predict([[34]]))

egitim_verisi=egitim_verisi.drop("date",axis=1)
egitim_verisi = egitim_verisi[['date_block_num','shop_id','item_id','item_price','item_category_id','item_cnt_day']]
egitim_verisi
inputs = egitim_verisi.iloc[:, :5]  
outputs = egitim_verisi.iloc[:, 5:6]  
print(inputs)
print(outputs)
TF_Modeli = Sequential()
TF_Modeli.add(Dense(32, input_dim=5, activation='relu'))   
TF_Modeli.add(Dropout(0.3)) 
TF_Modeli.add(Dense(1, activation='sigmoid'))   
TF_Modeli.compile(loss='mse', optimizer='adam', metrics=['accuracy'])   
TF_Modeli.summary()
TF_Modeli.fit(inputs, outputs, epochs=10)  
_, basari = TF_Modeli.evaluate(inputs, outputs)
print('Başarı Değeri: %.4f' % (basari*100))