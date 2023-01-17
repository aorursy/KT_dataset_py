import numpy as np

import pandas as pd



data = pd.read_excel('/kaggle/input/final_data.xlsx', parse_dates=True, index_col='date')

data=data[:-1]

data_raw=data.copy()



data.head()
data = data.interpolate(method='polynomial', order=1)



data.head()
data_usd = pd.read_csv('/kaggle/input/exchange_rate.csv')

data_usd_raw = data_usd.copy()



data_usd.head()
data_usd['date'] = pd.to_datetime(data_usd['date'],

                    format='%d.%m.%Y', errors='ignore')

data_usd = data_usd.set_index('date')

data_usd = data_usd['exrate']



data_usd.head()
import matplotlib.pyplot as plt

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()



plt.plot(data_usd)

plt.show()
data_usd = data_usd.resample('M').mean()



end_date = '2019-12-31'

data_usd = data_usd[:end_date]



data = data.assign(exrate = data_usd.values)



start_date = '2012-01-01'

data = data[start_date:]



data.head()
data_interbank = pd.read_excel('/kaggle/input/interbank.xlsx')



data_interbank.head()
import re

 

def regexp(reg):



    res = re.findall(r'\d{2}.\d{2}.\d{4}', reg)

    return res[0]     



data_interbank['date'] = data_interbank['date'].apply(regexp)



data_interbank['date'].head()
def replace(rep):



    rep = rep.replace(',', '.') 

    return rep



def to_float(fl):



    fl = float(fl)

    return fl



data_interbank['total_amount_usd'] = data_interbank['total_amount_usd'].apply(replace)

data_interbank['total_amount_usd'] = data_interbank['total_amount_usd'].apply(to_float)



data_interbank['total_amount_usd'].head()
data_interbank['date'] = pd.to_datetime(data_interbank['date'],

                    format='%d.%m.%Y', errors='ignore')

data_interbank = data_interbank.set_index('date')

data_interbank = data_interbank['total_amount_usd']

data_interbank = data_interbank.resample('M').sum()

data_interbank = data_interbank[start_date:]



data_interbank.head()



data = data.assign(interbank = data_interbank.values)
from sklearn.preprocessing import MinMaxScaler



scaler_X = MinMaxScaler(feature_range = (0, 1))



X = data.drop(labels=['exrate'], axis=1)

X = pd.DataFrame(scaler_X.fit_transform(X), columns = X.columns)



X.head()
scaler_y = MinMaxScaler(feature_range = (0, 1))



y = np.array(data['exrate'])

y = np.reshape(y, (len(y),-1))

y = pd.DataFrame(scaler_y.fit_transform(y))



y.head()
def raw_plot(data, column_name):



    plt.plot(data.index, data[column_name], label=column_name)

    plt.legend()

    plt.show()    

    

raw_plot(X, 'ppi')
def box(feat):



    plt.boxplot(x=X[feat])

    plt.title(feat)

    plt.show()     

    

box('ppi')
features = list(X.columns)

print(features)
def fix_outliers(column):

    

    learning_rate = 0.35

    

    q1 = X[column].quantile(0.25)

    q3 = X[column].quantile(0.75)

    iqr = q3-q1

    fence_low  = q1-1.5*iqr

    fence_high = q3+1.5*iqr

    

    X[column].loc[(X[column] >= fence_high)] = X[column].quantile(1-learning_rate)

    X[column].loc[(X[column] <= fence_low)] = X[column].quantile(learning_rate)

        

for col in features:

    fix_outliers(col)  
raw_plot(X, 'ppi')

box('ppi')
def feature_lag(features):



    for feature in features:

        X[feature + '-lag1'] = X[feature].shift(1)

        X[feature + '-lag2'] = X[feature].shift(2)

        X[feature + '-lag3'] = X[feature].shift(3)

        X[feature + '-lag6'] = X[feature].shift(6)

        X[feature + '-lag12'] = X[feature].shift(12)

    

feature_lag(features)  

X.drop(features, axis=1, inplace=True)



print(X.columns)
X.head()



real_X_size = len(X)

X = X.dropna()

dropna_X_size = len(X)

y = y[real_X_size-dropna_X_size:]
train_size = 0.78

separator = round(len(X.index)*train_size)



X_train, y_train = X.iloc[0:separator], y.iloc[0:separator]

X_test, y_test = X.iloc[separator:], y.iloc[separator:]
from keras.wrappers.scikit_learn import KerasRegressor

from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.optimizers import Adam

from keras.constraints import maxnorm



def build_model():



    model = Sequential([

    Dense(128, activation='relu', input_shape=[len(X.columns)],

                        kernel_constraint=maxnorm(5)),

    Dropout(0.3),

    Dense(1, kernel_initializer='normal', activation='sigmoid')])

    optimizer = Adam(lr=0.01)

    model.compile(optimizer=optimizer, loss='mean_squared_error',

                  metrics=['accuracy'])

    return model



model = KerasRegressor(build_fn=build_model, epochs=200, batch_size=10, verbose=0)
history = model.fit(X_train, y_train)

preds = model.predict(X_test)



print(preds)
predictions = scaler_y.inverse_transform([preds])

preds_real = [x for x in predictions[0]]



print(preds_real)
def predict_plot():



    ind_preds = data['2018-07-01':'2019-12-31']

    fig, axs = plt.subplots(1, figsize=(9,7))

    fig.suptitle('Predictions/real values')

    axs.plot(data.index, data.exrate, 'b-', label='real')

    axs.plot(ind_preds.index, preds_real, 'r-', label='prediction')

    axs.legend(loc=2)



predict_plot()  
from sklearn.metrics import mean_absolute_error



def errors(y_true, y_pred, r):

    

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    mape = np.mean(np.abs( (y_true - y_pred)/y_true))*100

    print('MAPE: {}%'.format(mape.round(r)))

    print('MAE: {}'.format(mean_absolute_error(y_true, y_pred).round(r)))  



errors(data_usd['2018-7-01':'2019-12-31'], preds_real, 3)