
import pandas as pd
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
df = pd.read_csv("../input/air-passengers/AirPassengers.csv")
df.head()
df["Month"] = pd.to_datetime(df["Month"])
df.head()

df.set_index("Month", inplace=True)
df.columns = ["passengers"]
df.index.name = "date"
df.head()
df.plot(figsize=(10,6))
from statsmodels.tsa.seasonal import seasonal_decompose
results = seasonal_decompose(df['passengers'])
results.plot();

len(df)
144-12
train = df.iloc[:132]
test = df.iloc[132:]

len(test)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
import warnings
warnings.filterwarnings("ignore")
scaler.fit(train)


scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)


from keras.preprocessing.sequence import TimeseriesGenerator
scaled_train
n_input = 2
n_features = 1
generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)
len(generator)
scaled_train
X,y = generator[0]
n_input = 12
generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)
scaled_train
X,y = generator[0]
print(f'Given the Array: \n{X.flatten()}')
print(f'Predict this y: \n {y}')


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_input, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.summary()
model.fit_generator(generator,epochs=50)
loss = model.history.history['loss']
plt.plot(range(len(loss)),loss)
first_eval_batch = scaled_train[-12:]
first_eval_batch
first_eval_batch = first_eval_batch.reshape((1, n_input, n_features))
model.predict(first_eval_batch)
scaled_test[0]
test_predictions = []

first_eval_batch = scaled_train[-n_input:]
current_batch = first_eval_batch.reshape((1, n_input, n_features))
current_batch.shape


current_batch
np.append(current_batch[:,1:,:],[[[99]]],axis=1)
test_predictions = []

first_eval_batch = scaled_train[-n_input:]
current_batch = first_eval_batch.reshape((1, n_input, n_features))

for i in range(len(test)):
    
    # get prediction 1 time stamp ahead ([0] is for grabbing just the number instead of [array])
    current_pred = model.predict(current_batch)[0]
    
    # store prediction
    test_predictions.append(current_pred) 
    
    # update batch to now include prediction and drop first value
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
test_predictions
scaled_test
true_predictions = scaler.inverse_transform(test_predictions)
true_predictions
test
test['Predictions'] = true_predictions
test
test.plot(figsize=(12,8))
model.save('rnn model')
from keras.models import load_model
new_model = load_model('rnn model')
new_model.summary()

