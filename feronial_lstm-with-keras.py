# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.layers import LSTM, Dense, Input

from keras.models import Model

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler





# Any results you write to the current directory are saved as output.
store_df = pd.read_csv('../input/stores data-set.csv')

features_df = pd.read_csv('../input/Features data set.csv')

main_df = pd.read_csv('../input/sales data-set.csv')
features_df = features_df.drop(columns = ['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5'])
features_df['Date'] = pd.to_datetime(features_df['Date'])

main_df['Date'] = pd.to_datetime(main_df['Date'])
main_df[main_df['Store'] ==1 ].plot.scatter(x = 'Dept', y = 'Weekly_Sales')
print(main_df.head(5))
print(features_df.head(5))
print(store_df.head(5))
main_df = main_df.merge(store_df)

main_df = main_df.merge(features_df)

main_df.head()
main_df.describe()
main_df.dtypes
main_df.info()
main_df['Label'] = main_df['Weekly_Sales']

main_df = main_df.drop(['Weekly_Sales'], axis = 1) #Sona taşı
main_df.head()
holiday_enc = LabelEncoder()

type_enc = LabelEncoder()





temp_minmax = MinMaxScaler()

fuel_minmax = MinMaxScaler()

cpi_minmax = MinMaxScaler()

unp_minmax = MinMaxScaler()

size_minmax = MinMaxScaler()





main_df['Temperature'] = temp_minmax.fit_transform(np.array(main_df['Temperature']).reshape(-1,1))

main_df['Fuel_Price'] = fuel_minmax.fit_transform(np.array(main_df['Fuel_Price']).reshape(-1,1))

main_df['CPI'] = cpi_minmax.fit_transform(np.array(main_df['CPI']).reshape(-1,1))

main_df['Unemployment'] = unp_minmax.fit_transform(np.array(main_df['Unemployment']).reshape(-1,1))

main_df['Size'] = size_minmax.fit_transform(np.array(main_df['Size']).reshape(-1,1))





main_df['IsHoliday'] = holiday_enc.fit_transform(main_df['IsHoliday'])

main_df['Type'] = type_enc.fit_transform(main_df['Type'])

main_df = main_df.sort_values(by = ['Store','Dept']) # LSTM'de sıralama 

main_df.head()
main_df = main_df.drop(['Date'], axis = 1)

#main_df = main_df[['Store','Dept','IsHoliday','Label']] #Sadece Tatil günleri etkisini hesaba katmak istenirse.

main_df = main_df[main_df['Store'] == 1] # 1. Dükkan için deneme

main_df = main_df[main_df['Dept'] == 1]
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):  # Kodu devşirdim. Yani sıfırdan kendi yazdığım bir kod değil. Veriyi kaydırarak(pd.shift) LSTM için uygun formata getiriyor.

    n_vars = 1 if type(data) is list else data.shape[1]

    df = pd.DataFrame(data)

    cols, names = list(), list()

    # Past

    for i in range(n_in, 0, -1): # n_in kadar kaydırma gerçekleştiriyor. [t, t-1, t-2 . . .] gibi.

        cols.append(df.shift(i))

        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

    # Future

    for i in range(0, n_out):

        cols.append(df.shift(-i))

        if i == 0:

            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]

        else:

            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

    # put it all together

    agg = pd.concat(cols, axis=1)

    agg.columns = names

    # drop rows with NaN values

    if dropnan:

        agg.dropna(inplace=True)

    return agg
lstm_df = series_to_supervised(main_df,3)
lstm_df = lstm_df.drop(lstm_df.columns[[-2,-3,-4,-5,-6,-7,-8,-9,-10]], axis = 1)

#lstm_df = lstm_df.drop(lstm_df.columns[[-2,-3,-4]], axis = 1) #Farklı geçiş değerleri için stünler değişkenlik gösterebilir.
np_lstm = np.array(lstm_df)
prop = 0.95



train_X, train_Y = np_lstm[:round(len(np_lstm) * prop), : np_lstm.shape[1] - 1 ], np_lstm[:round(len(np_lstm) * prop), -1 ]

test_X, test_Y = np_lstm[round(len(np_lstm) * prop):, : np_lstm.shape[1] - 1 ], np_lstm[round(len(np_lstm) * prop):, -1 ]



train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))

test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
inputs = Input(shape=(train_X.shape[1],train_X.shape[2]))



lstm_1 = LSTM(64, activation= 'relu')(inputs)



outputs = Dense(1, activation= 'relu')(lstm_1)



model = Model(inputs = inputs, outputs = outputs)

model.compile(loss='mean_absolute_percentage_error', 

              metrics=['mae'],

              optimizer='Adam')

model.summary()



model.fit(x = train_X,

         y=train_Y,

         batch_size = 2,

         epochs = 50,

         validation_data = (test_X, test_Y))
