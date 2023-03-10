import numpy as np # linear algebra

import pandas as pd # data processing

import matplotlib.pyplot as plt # data visualization



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# pip install yfinance

# import yfinance

# df = yf.download('BTC-USD','2017-01-02','2019-11-16')
df = pd.read_csv(r'/kaggle/input/bitcoin-usd-stock-prices/bitcoin.csv')
df.head()
df_close = pd.DataFrame(df['Close'])
df_close.index = pd.to_datetime(df['Date'])
df_close.index
df_close.head()
df_close.describe()
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()



plt.figure(figsize=(8, 6))

plt.plot(df_close, color='g')

plt.title('Bitcoin Closing Price', weight='bold', fontsize=16)

plt.xlabel('Time', weight='bold', fontsize=14)

plt.ylabel('USD ($)', weight='bold', fontsize=14)

plt.xticks(weight='bold', fontsize=12, rotation=45)

plt.yticks(weight='bold', fontsize=12)

plt.grid(color = 'y', linewidth = 0.5)
from statsmodels.tsa import stattools



acf_djia, confint_djia, qstat_djia, pvalues_djia = stattools.acf(df_close,

                                                             unbiased=True,

                                                             nlags=50,

                                                             qstat=True,

                                                             fft=True,

                                                             alpha = 0.05)



plt.figure(figsize=(7, 5))

plt.plot(pd.Series(acf_djia), color='r', linewidth=2)

plt.title('Autocorrelation of Bitcoin Closing Price', weight='bold', fontsize=16)

plt.xlabel('Lag', weight='bold', fontsize=14)

plt.ylabel('Value', weight='bold', fontsize=14)

plt.xticks(weight='bold', fontsize=12, rotation=45)

plt.yticks(weight='bold', fontsize=12)

plt.grid(color = 'y', linewidth = 0.5)
def create_regressor_attributes(df, attribute, list_of_prev_t_instants) :

    

    """

    Ensure that the index is of datetime type

    Creates features with previous time instant values

    """

        

    list_of_prev_t_instants.sort()

    start = list_of_prev_t_instants[-1] 

    end = len(df)

    df['datetime'] = df.index

    df.reset_index(drop=True)



    df_copy = df[start:end]

    df_copy.reset_index(inplace=True, drop=True)



    for attribute in attribute :

            foobar = pd.DataFrame()



            for prev_t in list_of_prev_t_instants :

                new_col = pd.DataFrame(df[attribute].iloc[(start - prev_t) : (end - prev_t)])

                new_col.reset_index(drop=True, inplace=True)

                new_col.rename(columns={attribute : '{}_(t-{})'.format(attribute, prev_t)}, inplace=True)

                foobar = pd.concat([foobar, new_col], sort=False, axis=1)



            df_copy = pd.concat([df_copy, foobar], sort=False, axis=1)

            

    df_copy.set_index(['datetime'], drop=True, inplace=True)

    return df_copy
list_of_attributes = ['Close']



list_of_prev_t_instants = []

for i in range(1,16):

    list_of_prev_t_instants.append(i)



list_of_prev_t_instants
df_new = create_regressor_attributes(df_close, list_of_attributes, list_of_prev_t_instants)

df_new.head()
df_new.shape
from tensorflow.keras.layers import Input, Dense, Dropout

from tensorflow.keras.optimizers import SGD

from tensorflow.keras.models import Model

from tensorflow.keras.models import load_model

from tensorflow.keras.callbacks import ModelCheckpoint
input_layer = Input(shape=(15), dtype='float32')

dense1 = Dense(60, activation='linear')(input_layer)

dense2 = Dense(60, activation='linear')(dense1)

dropout_layer = Dropout(0.2)(dense2)

output_layer = Dense(1, activation='linear')(dropout_layer)
model = Model(inputs=input_layer, outputs=output_layer)

model.compile(loss='mean_squared_error', optimizer='adam')

model.summary()
from tensorflow.keras.utils import plot_model

plot_model(model)
test_set_size = 0.05

valid_set_size= 0.05



df_copy = df_new.reset_index(drop=True)



df_test = df_copy.iloc[ int(np.floor(len(df_copy)*(1-test_set_size))) : ]

df_train_plus_valid = df_copy.iloc[ : int(np.floor(len(df_copy)*(1-test_set_size))) ]



df_train = df_train_plus_valid.iloc[ : int(np.floor(len(df_train_plus_valid)*(1-valid_set_size))) ]

df_valid = df_train_plus_valid.iloc[ int(np.floor(len(df_train_plus_valid)*(1-valid_set_size))) : ]





X_train, y_train = df_train.iloc[:, 1:], df_train.iloc[:, 0]

X_valid, y_valid = df_valid.iloc[:, 1:], df_valid.iloc[:, 0]

X_test, y_test = df_test.iloc[:, 1:], df_test.iloc[:, 0]



print('Shape of training inputs, training target:', X_train.shape, y_train.shape)

print('Shape of validation inputs, validation target:', X_valid.shape, y_valid.shape)

print('Shape of test inputs, test target:', X_test.shape, y_test.shape)
from sklearn.preprocessing import MinMaxScaler



Target_scaler = MinMaxScaler(feature_range=(0.01, 0.99))

Feature_scaler = MinMaxScaler(feature_range=(0.01, 0.99))



X_train_scaled = Feature_scaler.fit_transform(np.array(X_train))

X_valid_scaled = Feature_scaler.fit_transform(np.array(X_valid))

X_test_scaled = Feature_scaler.fit_transform(np.array(X_test))



y_train_scaled = Target_scaler.fit_transform(np.array(y_train).reshape(-1,1))

y_valid_scaled = Target_scaler.fit_transform(np.array(y_valid).reshape(-1,1))

y_test_scaled = Target_scaler.fit_transform(np.array(y_test).reshape(-1,1))
model.fit(x=X_train_scaled, y=y_train_scaled, batch_size=5, epochs=30, verbose=1, validation_data=(X_valid_scaled, y_valid_scaled), shuffle=True)
y_pred = model.predict(X_test_scaled)
y_pred_rescaled = Target_scaler.inverse_transform(y_pred)
from sklearn.metrics import r2_score

y_test_rescaled =  Target_scaler.inverse_transform(y_test_scaled)

score = r2_score(y_test_rescaled, y_pred_rescaled)

print('R-squared score for the test set:', round(score,4))
y_actual = pd.DataFrame(y_test_rescaled, columns=['Actual Close Price'])



y_hat = pd.DataFrame(y_pred_rescaled, columns=['Predicted Close Price'])
plt.figure(figsize=(11, 6))

plt.plot(y_actual, linestyle='solid', color='r')

plt.plot(y_hat, linestyle='dashed', color='b')



plt.legend(['Actual','Predicted'], loc='best', prop={'size': 14})

plt.title('Bitcoin Stock Closing Prices', weight='bold', fontsize=16)

plt.ylabel('USD ($)', weight='bold', fontsize=14)

plt.xlabel('Test Set Day no.', weight='bold', fontsize=14)

plt.xticks(weight='bold', fontsize=12, rotation=45)

plt.yticks(weight='bold', fontsize=12)

plt.grid(color = 'y', linewidth='0.5')

plt.show()