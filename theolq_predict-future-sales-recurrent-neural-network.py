import pandas as pd

train = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')

items = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')

test = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')
train.head(10)
import matplotlib.pyplot as plt

import seaborn as sns; sns.set()



plt.figure(figsize=(10,4))

plt.xlim(-100, 3000)

sns.boxplot(x=train.item_cnt_day)

plt.title("Outliers detection on item_cnt_day")

plt.show()



plt.figure(figsize=(10,4))

plt.xlim(train.item_price.min(), train.item_price.max()*1.1)

sns.boxplot(x=train.item_price)

plt.title("Outliers detection on item_price")

plt.show()
train = train[train.item_price <= 100000]

train = train[train.item_cnt_day <= 1000]
below_zero = train[train["item_price"] <= 0]

below_zero.head()
shop_32 = train[ (train.shop_id==32) & (train.item_id==2973)]

shop_32.head(10)
plt.plot(shop_32["item_price"], 'o')

plt.title("Price variation of the item")

plt.show()
value = train[(train["shop_id"] == 32) & (train["item_id"] == 2973) & (train["date_block_num"] == 4) & (train["item_price"] > 0)]["item_price"].median()

train.loc[train["item_price"] < 0, 'item_price'] = value
#Якутск Орджоникидзе, 56

train.loc[train["shop_id"] == 0, 'shop_id'] = 57

test.loc[test["shop_id"] == 0, 'shop_id'] = 57



#Якутск ТЦ "Центральный"

train.loc[train["shop_id"] == 1, 'shop_id'] = 58

test.loc[test["shop_id"] == 1, 'shop_id'] = 58



#Жуковский ул. Чкалова 39м²

train.loc[train["shop_id"] == 10, 'shop_id'] = 11

test.loc[test["shop_id"] == 10, 'shop_id'] = 11
shops = pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv')

shops.head(10)
shops['city'] = shops['shop_name'].str.split(' ').map(lambda x: x[0])

shops = shops[['shop_id','city']]

shops.head(10)
categorie = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')

categorie.head(10)
def extract_subtype(string):

    if len(string) == 2:

        return string[1].strip()

    else:

        return string[0]



categorie['type'] = categorie['item_category_name'].str.split('-').map(lambda x: x[0].strip())

categorie['sub_type'] = categorie['item_category_name'].str.split('-').map(lambda x: extract_subtype(x))

categorie = categorie[['item_category_id','type', 'sub_type']]

categorie.head(10)
train['date'] = pd.to_datetime(train['date'], format='%d.%m.%Y')

train.head(10)
df = train.pivot_table(index=['item_id', 'shop_id'], values=['item_cnt_day'], columns='date_block_num', fill_value=0)

df.head(10)
df = pd.merge(test, df, on=['item_id', 'shop_id'], how='left')

df = df.fillna(0)

df.head()
for i in range(34):

    df[('item_cnt_day', i)].clip(0, 20)
df = pd.merge(df, shops, on=["shop_id"], how='left')

df.head()
items = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')

items = items.drop("item_name", axis=1)

items.head()
df = pd.merge(df, items, on=["item_id"], how='left')

df.head()
df = pd.merge(df, categorie, on=["item_category_id"], how='left')

df.head()
df = df.drop(['shop_id', 'item_id', 'ID', 'item_category_id'], axis=1)

df.head()
df = pd.get_dummies(df, columns=["city", "type", "sub_type"])

df.head(10)
import tensorflow as tf

from tensorflow import keras

import numpy as np





X_time_series = df.iloc[:, :34]

X_city = df.iloc[:, 34:62]

X_type = df.iloc[:, 63:]



X_train_series = np.expand_dims(X_time_series.values[:, :-1], axis=2)

X_train_city = np.expand_dims(X_city.values, axis=2)

X_train_type = np.expand_dims(X_type.values, axis=2)

y_train = X_train_series[:, -1:]



X_test_series = np.expand_dims(X_time_series.values[:, 1:], axis=2)

X_test_city = np.expand_dims(X_city.values, axis=2)

X_test_type = np.expand_dims(X_type.values, axis=2)
input_time_series = keras.layers.Input(shape=(33, 1))

input_city = keras.layers.Input(shape=(28, 1))

input_type = keras.layers.Input(shape=(64, 1))





#WaveNet

wave_1 = keras.layers.Conv1D(filters=16, kernel_size=2, padding="causal", dilation_rate=1, kernel_initializer="glorot_normal")(input_time_series)

BN_1 = keras.layers.BatchNormalization()(wave_1)

relu_1 = keras.layers.Activation("relu")(BN_1)



wave_2 = keras.layers.Conv1D(filters=32, kernel_size=2, padding="causal", dilation_rate=2, kernel_initializer="glorot_normal")(relu_1)

BN_2 = keras.layers.BatchNormalization()(wave_2)

relu_2 = keras.layers.Activation("relu")(BN_2)



wave_3 = keras.layers.Conv1D(filters=64, kernel_size=2, padding="causal", dilation_rate=4, kernel_initializer="glorot_normal")(relu_2)

BN_3 = keras.layers.BatchNormalization()(wave_3)

relu_3 = keras.layers.Activation("relu")(BN_3)



wave_4 = keras.layers.Conv1D(filters=128, kernel_size=2, padding="causal", dilation_rate=8, kernel_initializer="glorot_normal")(relu_3)

BN_4 = keras.layers.BatchNormalization()(wave_4)

relu_4 = keras.layers.Activation("relu")(BN_4)



before_concat = keras.layers.Conv1D(filters=256, kernel_size=1, kernel_initializer="glorot_normal")(relu_4)

before_concat_BN = keras.layers.BatchNormalization()(before_concat)

before_concat_relu = keras.layers.Activation("relu")(before_concat_BN)





#Flatten

flattened_time_series = keras.layers.Flatten()(before_concat_relu)

flattened_city = keras.layers.Flatten()(input_city)

flattened_product = keras.layers.Flatten()(input_type)

concat = keras.layers.concatenate([flattened_time_series, flattened_city])





#First Hidden

hidden_1 = keras.layers.Dense(512, kernel_initializer="glorot_normal")(concat)

hidden_BN_1 = keras.layers.BatchNormalization()(hidden_1)

hidden_relu_1 = keras.layers.Activation("relu")(hidden_BN_1)



hidden_2 = keras.layers.Dense(512, kernel_initializer="glorot_normal")(concat)

hidden_BN_2 = keras.layers.BatchNormalization()(hidden_2)

hidden_relu_2 = keras.layers.Activation("relu")(hidden_BN_2)





concat_2 = keras.layers.concatenate([hidden_relu_2, flattened_product])





#Second Hidden

hidden_3 = keras.layers.Dense(256, kernel_initializer="glorot_normal")(concat_2)

hidden_BN_3 = keras.layers.BatchNormalization()(hidden_3)

hidden_relu_3 = keras.layers.Activation("relu")(hidden_BN_3)



hidden_4 = keras.layers.Dense(256, kernel_initializer="glorot_normal")(hidden_relu_3)

hidden_BN_4 = keras.layers.BatchNormalization()(hidden_4)

hidden_relu_4 = keras.layers.Activation("relu")(hidden_BN_4)



#Third Hidden



hidden_5 = keras.layers.Dense(128, kernel_initializer="glorot_normal")(hidden_relu_4)

hidden_BN_5 = keras.layers.BatchNormalization()(hidden_5)

hidden_relu_5 = keras.layers.Activation("relu")(hidden_BN_5)

dropout_1 = keras.layers.Dropout(rate=0.5)(hidden_relu_5)



hidden_6 = keras.layers.Dense(64, kernel_initializer="glorot_normal")(dropout_1)

hidden_BN_6 = keras.layers.BatchNormalization()(hidden_6)

hidden_relu_6 = keras.layers.Activation("relu")(hidden_BN_6)

dropout_2 = keras.layers.Dropout(rate=0.5)(hidden_relu_6)



hidden_7 = keras.layers.Dense(32, kernel_initializer="glorot_normal")(dropout_2)

hidden_BN_7 = keras.layers.BatchNormalization()(hidden_7)

hidden_relu_7 = keras.layers.Activation("relu")(hidden_BN_7)

dropout_3 = keras.layers.Dropout(rate=0.5)(hidden_relu_7)





#End

output = keras.layers.Dense(1)(dropout_3)



model = keras.models.Model(inputs=[input_time_series, input_city, input_type], outputs=[output])

model.compile(loss="mse", optimizer=keras.optimizers.SGD(momentum=0.9), metrics=['mae'])

model.summary()
callbacks = [

    keras.callbacks.EarlyStopping(patience=5),

    keras.callbacks.ModelCheckpoint("model.h5", save_best_only=True),

]



history = model.fit([X_train_series, X_train_city, X_train_type], y_train, batch_size=256, epochs=100, callbacks=callbacks)
import matplotlib.pyplot as plt

import seaborn as sns; sns.set()



def plot_learning_curves(history):

    pd.DataFrame(history.history).plot(figsize=(8, 5))

    plt.gca().set_ylim(0, 1)

    plt.show()

    

plot_learning_curves(history)
prediction = model.predict([X_test_series, X_test_city, X_test_type])

prediction = prediction.clip(0, 20)
from os import chdir

chdir('../working')



submission = pd.DataFrame({'ID': test['ID'], 'item_cnt_month': prediction.ravel()})

submission.to_csv('submission.csv',index=False)