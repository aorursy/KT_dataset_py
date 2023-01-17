#gerekli kutuphanelerin eklenmesi
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import andrews_curves
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
model_lstm = Sequential()
model_lstm.add(LSTM(15, input_shape=(1,6)))
model_lstm.add(Dense(1))
model_lstm.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
#csv dosyalarini okuma https://www.kaggle.com/getting-started/25930 linkinden yararlanilmistir
import os
def print_files():
    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames:
            print(os.path.join(dirname, filename))
#csv dosyalari
print_files()
# Ilk olarak okunacak dosyanin yolu belirlenir. Satislarin bulundugu dosyadan verileri okuyarak tahmin etme islemi gerceklestirmeye calisiyorum.

PATH = "../input/competitive-data-science-predict-future-sales/sales_train.csv"

sales_train = pd.read_csv(PATH)
sales_train.head() #dosyadaki degerlerin gosterilmesi icin eklenmistir
print('Train:', sales_train.shape)
sample_submission = pd.read_csv('../input/competitive-data-science-predict-future-sales/sample_submission.csv')
sample_submission.head()
test = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')
test.head()
# kategori id de en buyuk degeri buluyorum
# path e items.csv verilir
PATH = '../input/competitive-data-science-predict-future-sales/items.csv'
items = pd.read_csv(PATH)
print('id(max) : ', items['item_category_id'].max())

items.head()
sales_train.head()
train_clean = sales_train.drop(labels = ['date', 'item_price'], axis = 1)
train_clean.head()
train_clean = train_clean.groupby(["item_id","shop_id","date_block_num"]).sum().reset_index()
train_clean = train_clean.rename(index=str, columns = {"item_cnt_day":"item_cnt_month"})
train_clean = train_clean[["item_id","shop_id","date_block_num","item_cnt_month"]]
train_clean
check = train_clean[["shop_id","item_id","date_block_num","item_cnt_month"]]
check = check.loc[check['shop_id'] == 55]
check = check.loc[check['item_id'] == 1]
check
plt.figure(figsize=(10,4))
plt.title('Item 1 - Shop 55 Aylık Satıs Gosterimi')
plt.xlabel('Month')
plt.ylabel('Item 1 - Shop 55 Satis Miktari')
plt.plot(check["date_block_num"],check["item_cnt_month"]);
num_month = sales_train['date_block_num'].max()

month_list=[i for i in range(num_month+1)]
shop = []
for i in range(num_month+1):
    shop.append(55)
    
item = []
for i in range(num_month+1):
    item.append(1)
    
months_full = pd.DataFrame({'shop_id':shop, 'item_id':item,'date_block_num':month_list})
months_full
sales_33month = pd.merge(check, months_full, how='right', on=['shop_id','item_id','date_block_num'])
sales_33month = sales_33month.sort_values(by=['date_block_num'])
sales_33month.fillna(0.00,inplace=True)
sales_33month
plt.figure(figsize=(10,4))
plt.title('Item 1 - Shop 55 Aylık Tüm Satışları')
plt.xlabel('Aylar')
plt.ylabel('Item 1 - Shop 55 Satışlar')
plt.plot(sales_33month["date_block_num"],sales_33month["item_cnt_month"]);
for i in range(1,4):
    sales_33month["Train" + str(i)] = sales_33month.item_cnt_month.shift(i)
sales_33month.fillna(0.0, inplace=True)
sales_33month
df = sales_33month[['shop_id','item_id','date_block_num','Train1','Train2','Train3', 'item_cnt_month']].reset_index()
df = df.drop(labels = ['index'], axis = 1)
df
train_df = df[:-3]
val_df = df[-3:]
x_train,y_train = train_df.drop(["item_cnt_month"],axis=1),train_df.item_cnt_month
x_val,y_val = val_df.drop(["item_cnt_month"],axis=1),val_df.item_cnt_month
x_train
y_train
x_val
y_val
from sklearn.preprocessing import StandardScaler,MinMaxScaler
scaler = StandardScaler()
scaler = MinMaxScaler(feature_range=(-1, 1))
x_train_scaled = scaler.fit_transform(x_train)
x_valid_scaled = scaler.fit_transform(x_val)
x_train_reshaped = x_train_scaled.reshape((x_train_scaled.shape[0], 1, x_train_scaled.shape[1]))
x_val_resaped = x_valid_scaled.reshape((x_valid_scaled.shape[0], 1, x_valid_scaled.shape[1]))
history = model_lstm.fit(x_train_reshaped, y_train, validation_data=(x_val_resaped, y_val),epochs=100, batch_size=12, verbose=2, shuffle=False)
y_pre = model_lstm.predict(x_val_resaped)
fig, ax = plt.subplots()
ax.plot(x_val['date_block_num'], y_val, label='Actual')
ax.plot(x_val['date_block_num'], y_pre, label='Predicted')
plt.title('LSTM Prediction vs Actual - 3 Aylık Satışlar')
plt.xlabel('Ay')
plt.xticks(x_val['date_block_num'])
plt.ylabel('Item 1 - Shop 55 Satışları')
ax.legend()
plt.show()
from sklearn.metrics import mean_squared_error
from numpy import sqrt
rmse = sqrt(mean_squared_error(y_val,y_pre))
print('Val RMSE: %.3f' % rmse)