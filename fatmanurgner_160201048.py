# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd


train = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')
test = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')
items = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')
item_category = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')
shop = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')

train.head()

items.head()
test.head()
print(train.shape, test.shape)
##Train datasındaki tarihler text verisi oldğu için ppython da 
## kullanmak adına datetime objelerine çevirmemiz gerekiyor

train['date'] = pd.to_datetime(train['date'], format='%d.%m.%Y')
train.head()
Veriler = train.pivot_table(index=['item_id', 'shop_id'],values=['item_cnt_day'], columns='date_block_num', fill_value=0)
Veriler = Veriler.reset_index()
Veriler = pd.merge(test, Veriler, on=['item_id', 'shop_id'], how='left')
Veriler = Veriler.fillna(0)

Veriler.head()

## Shop_id ve item_id değerleri için tekil bir ID bilgisi olusturuluyor.
##Temelde test ve train datas harmanlanıyor gibi düşünebiliriz

Veriler = Veriler.drop(['shop_id', 'item_id', 'ID'], axis=1)
Veriler.head()
X_train = np.expand_dims(Veriler.values[:, :-1], axis=2)
y_train = Veriler.values[:, -1:]

X_test = np.expand_dims(Veriler.values[:, 1:], axis=2)
print(X_train.shape, y_train.shape, X_test.shape)
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
tahmin = Sequential()
tahmin.add(LSTM(64, input_shape=(33, 1)))
tahmin.add(Dense(20, input_dim=2, activation='relu'))
tahmin.add(Dropout(0.3))
tahmin.add(Dense(1, activation='sigmoid'))

tahmin.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model_fitting = tahmin.fit(X_train, y_train, batch_size=4096, epochs=5)
xx, yy = np.meshgrid(np.arange(-2, 3, 0.1),
                     np.arange(-1.5, 2, 0.1))


from keras.layers import Input, Dense, Dropout
from keras.models import Model
import matplotlib.pyplot as plt
_, accuracy=tahmin.evaluate(X_train, y_train)

model_prediction = tahmin.predict(X_test)
model_prediction = model_prediction.clip(0, 20)

submission = pd.DataFrame({'ID': test['ID'], 'item_cnt_month': model_prediction.ravel()})
submission.head()
print('Modelin Elde Ettiği Doğruluk Oranıı:%.2f'%(accuracy*100))