# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import pandas as pd

import datetime

import numpy as np

import matplotlib.pyplot as plt





from sklearn import metrics

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

import operator



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
items = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')

train = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')

test  = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')
items.head() # Ilk 5 satiri gosteriyoruz.
items.info() # 22170 item var. Null degere sahip olan yok
train.head() # Ilk 5 satiri gosteriyoruz.
train['item_cnt_day'] = train['item_cnt_day'].astype(np.int16) # item_cnt_day degerlerini integer yapiyoruz
train.head()
test.head() # Ilk 5 satiri gosteriyoruz.
test_shops = test.shop_id.unique() # Test verisi icindeki unique shop_id degerleri aliyoruz

train = train[train.shop_id.isin(test_shops)] # Train verisi icinde sadece test verisi icinde olan shop_id'leri birakiyoruz

test_items = test.item_id.unique() # Test verisi icindeki unique item_id degerlerini aliyoruz.

train = train[train.item_id.isin(test_items)] # Train verisi icinde sadece test verisi icinde olan item_id'leri birakiyoruz
train.sort_values(by="item_cnt_day")
train.loc[train.item_cnt_day < 0, "item_cnt_day"] = 0 # Satis miktari 0'dan kucuk olanlari 0'a esitliyoruz.
train.sort_values(by=["item_cnt_day"]).head(2)
train.sort_values(by=["item_price"]).head(2) # Fiyati 0'dan dusuk olan var mi diye kontrol ediyoruz.
train[train.duplicated()]
train.loc[train.date_block_num == 13].loc[train.shop_id == 50].loc[train.item_id == 3423].loc[train.date == "23.02.2014"]
len(train[train.duplicated()]) # Duplicate veri sayisi : 5
train.drop_duplicates(inplace=True) # Duplicate verileri drop ediyoruz.
train.reset_index(inplace=True)

train.drop(["index"], axis=1, inplace=True)
train.info()
dropped = train.drop(["item_price", "date"], axis=1) # Kullanilmayacagini dusundugum sutunlari drop ediyorum.
dropped.head()
grouped_train = dropped.groupby(["date_block_num","shop_id","item_id"]).sum().reset_index() # Her ay toplam satilmis esya miktarini topluyoruz.
grouped_train.tail()
grouped_train.loc[grouped_train.item_cnt_day > 100].sample(10)
grouped_train.loc[grouped_train['item_cnt_day'] == grouped_train["item_cnt_day"].max()] # Bir ayda maximum satisi yapilmis urun
all_data_list = []



for i in range(34):

  for shop in grouped_train.shop_id.unique():

    for item in grouped_train.item_id.unique():

      all_data_list.append([i, shop, item])



all_data_df = pd.DataFrame(all_data_list, columns=["date_block_num", "shop_id", "item_id"])
all_data_df
train_monthly = pd.merge(all_data_df, grouped_train, on=['date_block_num','shop_id','item_id'], how='left')

train_monthly.fillna(0, inplace=True)
train_monthly
train_monthly['item_cnt_day'] = train_monthly['item_cnt_day'].astype(np.int16)
train_monthly.rename(columns={"item_cnt_day": "item_cnt_month"},inplace=True)
train_monthly
train_monthly.loc[train_monthly.shop_id == 21].loc[train_monthly.item_id == 20949]
testDF = train_monthly.loc[train_monthly.shop_id == 21].loc[train_monthly.item_id == 20949].reset_index()

testDF.drop(["index"], axis=1, inplace=True)
x = testDF.date_block_num

y = testDF.item_cnt_month



x = x[:, np.newaxis]

y = y[:, np.newaxis]



polynomial_features= PolynomialFeatures(degree=6)

x_poly = polynomial_features.fit_transform(x)



model = LinearRegression()

model.fit(x_poly, y)

y_poly_pred = model.predict(x_poly)



rmse = np.sqrt(mean_squared_error(y,y_poly_pred))

r2 = r2_score(y,y_poly_pred)

print("rmse degeri",rmse)

print("r2 degeri",r2)



plt.scatter(x, y, s=10)

sort_axis = operator.itemgetter(0)

sorted_zip = sorted(zip(x,y_poly_pred), key=sort_axis)

x, y_poly_pred = zip(*sorted_zip)

plt.plot(x, y_poly_pred, color='m')

plt.show()