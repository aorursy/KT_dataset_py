import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.models import load_model
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error, r2_score
K.clear_session()
df = pd.read_csv('../input/stock-market-prediction/RELIANCE.NS(1).csv')
df.head()
df.shape
df.isna().sum()
df = df.dropna()
df.isna().sum()
df.shape
df.describe()
correlation = df.corr()
correlation
f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(correlation, annot=True, linewidths=.5, fmt= '.1f',ax=ax)
X=df.drop(['Adj Close','Close'],axis=1)
X.corrwith(df['Adj Close']).plot.bar(
        figsize = (15, 5), title = "Correlation with Adj Close", fontsize = 20,
        rot = 90, grid = True)
#very simple plotting
f,ax1 = plt.subplots(figsize=(15, 5))
ax1.set_ylabel('Price')
ax1.set_title('Original Plot')
ax1.plot('Adj Close', data = df);
df.plot(figsize=(23,8),title = "Reliance Stock Price Analysis")
plt.subplot(411)
plt.plot(df.Open, label='Open')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(df.Low, label='Low')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(df.Close,label='High')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(df['Close'], label='Adj Close')
plt.legend(loc='best')
plt.tight_layout()
plt.show()
# Feature Selection
X = ['Open', 'High', 'Low', 'Volume']
y = pd.DataFrame(df['Adj Close'])
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
feature_minmax_transform_data = scaler.fit_transform(df[X])
feature_minmax_transform = pd.DataFrame(columns=X, data=feature_minmax_transform_data, index=df.index)
feature_minmax_transform.head()
display(feature_minmax_transform.head())
print('Shape of features : ', feature_minmax_transform.shape)
print('Shape of target : ', y.shape)

# Shift target array because we want to predict the n + 1 day value


target_adj_close = y.shift(-1)
validation_y = y[-90:-1]
target_adj_close = y[:-90]

# Taking last 90 rows of data to be validation set
validation_X = feature_minmax_transform[-90:-1]
feature_minmax_transform = feature_minmax_transform[:-90]
display(validation_X.tail())
display(validation_y.tail())

print("\n -----After process------ \n")
print('Shape of features : ', feature_minmax_transform.shape)
print('Shape of target : ', y.shape)
display(y.tail())
from sklearn.model_selection import TimeSeriesSplit
ts_split = TimeSeriesSplit(n_splits=10)
for train_index, test_index in ts_split.split(feature_minmax_transform):
        X_train, X_test = feature_minmax_transform[:len(train_index)], feature_minmax_transform[len(train_index): (len(train_index)+len(test_index))]
        y_train, y_test = y[:len(train_index)].values.ravel(), y[len(train_index): (len(train_index)+len(test_index))].values.ravel()
len(X_train), len(X_test), len(y_train), len(y_test)
X_train =np.array(X_train)
X_test =np.array(X_test)

X_tr_t = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_tst_t = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
model_lstm = Sequential()
model_lstm.add(LSTM(100, input_shape=(1, X_train.shape[1]), activation='relu', return_sequences=False))
model_lstm.add(Dense(1))
model_lstm.compile(loss='mean_squared_error', optimizer='adam')
early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)
history_model_lstm = model_lstm.fit(X_tr_t, y_train, epochs=600, batch_size=10, verbose=1, shuffle=False)
y_pred_test_lstm = model_lstm.predict(X_tst_t)
y_train_pred_lstm = model_lstm.predict(X_tr_t)
print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train, y_train_pred_lstm)))
r2_train = r2_score(y_train, y_train_pred_lstm)

print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_pred_test_lstm)))
r2_test = r2_score(y_test, y_pred_test_lstm)
y_pred_test_LSTM = model_lstm.predict(X_tst_t)
plt.plot(y_test, label='True')
plt.plot(y_pred_test_lstm, label='LSTM')
plt.title("LSTM's_Prediction")
plt.xlabel('Observation')
plt.ylabel('INR_Scaled')
plt.legend()
plt.show()
col1 = pd.DataFrame(y_test, columns=['True'])

col2 = pd.DataFrame(y_pred_test_LSTM, columns=['LSTM_prediction'])

col3 = pd.DataFrame(history_model_lstm.history['loss'], columns=['Loss_LSTM'])
results = pd.concat([col1, col2, col3], axis=1)
results.head()
