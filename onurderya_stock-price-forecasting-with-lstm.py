import pandas as pd 
df = pd.read_csv('../input/apple-stock-price-20132017/Dataset.csv')

df.head()
df.plot(kind='line', x='Date', y='Open', figsize=(15,5)) 
import numpy as np              

import matplotlib.pyplot as plt 

import pandas as pd           
apple_traning_dataset = pd.read_csv('../input/apple-stock-price-20132017/Dataset.csv')
apple_traning_dataset.head()

apple_traning_dataset.info()
apple_traning_processed = apple_traning_dataset.iloc[:, 1:2].values
apple_traning_processed
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

apple_traning_scaled = scaler.fit_transform(apple_traning_processed)

features_set = [] #input
labels = [] #output
for i in range(60, 1259):
    features_set.append(apple_traning_scaled[i-60:i, 0])
    labels.append(apple_traning_scaled[i, 0])

features_set, labels = np.array(features_set), np.array(labels) 
#son değişkenlerimizi numpy dizisine çeviriyoruz
features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1))
#Keras kütüphanesi arkaplanda TensorFlow'u kullanacaktır. 
from keras.models import Sequential 
from keras.layers import Dense 
from keras.layers import LSTM  
from keras.layers import Dropout 
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(features_set.shape[1], 1)))
model.add(Dropout(0.2)) 
# 3 tane daha LSTM modeli ve dropout katmanı ekliyoruz.
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units = 1))
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.fit(features_set, labels, epochs = 100, batch_size = 32)
apple_testing_complete = pd.read_csv('../input/apple-stock-price-20132017/Dataset_testing.csv')
apple_testing_processed = apple_testing_complete.iloc[:, 1:2].values
df_testing = pd.read_csv('../input/apple-stock-price-20132017/Dataset_testing.csv')
df_testing.head()
df_testing.plot(kind='line', x='Date', y='Open', figsize=(15,5))
apple_total = pd.concat((apple_traning_dataset['Open'], apple_testing_complete['Open']), axis=0)
test_inputs = apple_total[len(apple_total) - len(apple_testing_complete) - 60:].values
test_inputs
test_inputs = test_inputs.reshape(-1,1)
test_inputs = scaler.transform(test_inputs)
test_features = []
for i in range(60, 80):
    test_features.append(test_inputs[i-60:i, 0])
test_features = np.array(test_features)
test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1], 1))
predictions = model.predict(test_features)
predictions = scaler.inverse_transform(predictions)
plt.figure(figsize=(15,5))
plt.plot(apple_testing_processed, color='blue', label='Reel Apple Hisse Senedi')
plt.plot(predictions , color='red', label='Tahmini Apple Hisse Senedi')
plt.title('Apple Hisse Senedi Tahminlemesi')
plt.xlabel('Date')
plt.ylabel('Apple Stock Price')
plt.legend()
plt.show()