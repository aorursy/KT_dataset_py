#import libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

%matplotlib inline
#load data

test = pd.read_csv("../input/mobile-price-classification/test.csv")

train = pd.read_csv("../input/mobile-price-classification/train.csv")
train.head()
test.head()
train.info()
test.info()
numerical = ['battery_power','clock_speed','fc','int_memory','m_dep','mobile_wt', 'n_cores', 'pc', 'px_height',

             'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time']

categorical = ['blue','dual_sim','four_g','three_g', 'touch_screen','wifi']
print(len(numerical))

print(len(categorical))
#categorical attributes

df = pd.melt(train[categorical])

sns.countplot(data=df,x='variable', hue='value')
#numerical attributes

fig = plt.figure(figsize=(15,20))

for i,col in enumerate(numerical):

    ax=plt.subplot(5,3,i+1) 

    train[col].plot.hist(ax = ax).tick_params(axis = 'x',labelrotation = 360)

    ax.legend(loc = 'upper center', bbox_to_anchor=(0.5, 1.1))

plt.show()
skewed = ['clock_speed','fc','m_dep', 'px_height', 'sc_w']

no_skewed = ['battery_power','int_memory','mobile_wt','n_cores','pc','px_width','ram','sc_h','talk_time']
#correlation between attributes

corr = train.corr()

fig, (ax) = plt.subplots(1,1,sharey = True, figsize = (20,10))

sns.heatmap(corr, cmap = 'Blues')
#correlation between price and phone attributes

corr.sort_values(by=["price_range"],ascending=False).iloc[0].sort_values(ascending=False)
train.groupby('price_range').mean()['ram'].plot(kind = 'bar', legend = True).tick_params(axis = 'x', labelrotation = 360)
#variables with symmetrical distributions

group_no_skewed = train.groupby('price_range')[no_skewed].mean().reset_index()

fig = plt.figure(figsize=(15,20))

for i,col in enumerate(group_no_skewed.iloc[:,1:].columns):

    ax=plt.subplot(5,3,i+1) 

    group_no_skewed.iloc[:,1:][col].plot.bar(ax = ax).tick_params(axis = 'x',labelrotation = 360)

    ax.legend(loc = 'upper center', bbox_to_anchor=(0.5, 1.1))

plt.show()
#variables with skewed distributions

group_skewed = train.groupby('price_range')[skewed].median().reset_index()

fig = plt.figure(figsize=(15,20))

for i,col in enumerate(group_skewed.iloc[:,1:].columns):

    ax=plt.subplot(5,3,i+1) 

    group_skewed.iloc[:,1:][col].plot.bar(ax = ax).tick_params(axis = 'x',labelrotation = 360)

    ax.legend(loc = 'upper center', bbox_to_anchor=(0.5, 1.1))

plt.show()
#bluetooth, wifi vs. price

sns.catplot('price_range', col='blue',hue = 'wifi',data = train,  kind = 'count', col_wrap=2)
#3g, 4g vs. price

sns.catplot('price_range', col='three_g',hue = 'four_g',data = train,  kind = 'count', col_wrap=2)
#dual_sim vs. price

sns.catplot('price_range', col='dual_sim',data = train,  kind = 'count')
#touch_screen vs. price

sns.catplot('price_range', col='touch_screen',data = train,  kind = 'count')
#scale numeric variables of training data

from sklearn.preprocessing import MinMaxScaler

scaler_train = MinMaxScaler()

train_num_scaled = scaler_train.fit_transform(train[numerical])

scaler_train.data_max_

scaler_train.data_min_
train_num_scaled = pd.DataFrame(train_num_scaled,columns=train[numerical].columns)

train_num_scaled
#scale numeric variables of test data

from sklearn.preprocessing import MinMaxScaler

scaler_test = MinMaxScaler()

test_num_scaled = scaler_test.fit_transform(test[numerical])

scaler_test.data_max_

scaler_test.data_min_
test_num_scaled = pd.DataFrame(test_num_scaled,columns=test[numerical].columns)
test_final = pd.concat([test[categorical],test_num_scaled], axis = 1)

test_final.head()
#X & Y array

import tensorflow as tf

X = pd.concat([train[categorical],train_num_scaled], axis = 1)

y = tf.keras.utils.to_categorical(train['price_range'], 4)
X.head()
y
#Split the original train data into train and val data

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, train['price_range'], test_size=0.33, random_state=101)
print(X_train.shape)

print(y_train.shape)

print(X_val.shape)

print(y_val.shape)
#import deep learning libraries

import tensorflow.keras

from keras.models import Sequential

from keras.layers import Dense
#build model

model_1 = Sequential()

model_1.add(Dense(25, input_dim=20, activation='relu'))

model_1.add(Dense(25, activation='relu'))

model_1.add(Dense(4, activation='softmax'))

model_1.summary()
model_1.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])

hist_1 = model_1.fit(X_train, y_train, epochs=20, batch_size=25, 

                   validation_data=(X_val,y_val))
plt.plot(hist_1.history['loss'])

plt.plot(hist_1.history['val_loss'])



plt.title('Model Loss Progression During Training/Validation')

plt.ylabel('Training and Validation Losses')

plt.xlabel('Epoch Number')

plt.legend(['Training Loss', 'Validation Loss'])
score = model_1.evaluate(X_val, y_val, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
#test data prediction

prediction_test = np.argmax(model_1.predict(test_final), axis=1)

pd.DataFrame({'id' : test['id'],'price_range' : prediction_test})