import pandas as pd

train = pd.read_csv('../input/nyc-taxi-trip-duration/train.zip')

test = pd.read_csv('../input/nyc-taxi-trip-duration/test.zip')

submission = pd.read_csv('../input/nyc-taxi-trip-duration/sample_submission.zip')
train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'])

train['dropoff_datetime'] = pd.to_datetime(train['dropoff_datetime'])

test['pickup_datetime'] = pd.to_datetime(test['pickup_datetime'])
train['pickup_hour'] = train['pickup_datetime'].dt.hour

test['pickup_hour'] = test['pickup_datetime'].dt.hour
columns = ['passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'pickup_hour']

X_train = train[columns]

Y_train = train['trip_duration']

X_test = test[columns]
X_train.shape, Y_train.shape, X_test.shape
X_train_norm = (X_train - X_train.mean()) / X_train.std()

Y_train_norm = (Y_train - Y_train.mean()) / Y_train.std()

X_test_norm = (X_test - X_test.mean()) / X_test.std()
X_train_norm.mean()
X_train_norm.std()
Y_train_norm.mean()
Y_train_norm.std()
X_test_norm.mean()
X_test_norm.std()
import tensorflow as tf
model = tf.keras.models.Sequential() 

model.add(tf.keras.layers.Dense(128, activation = 'relu', input_shape = (6,) ))

model.add(tf.keras.layers.Dense(128, activation = 'relu'))

model.add(tf.keras.layers.Dense(128, activation = 'relu'))

model.add(tf.keras.layers.Dense(64, activation = 'relu'))

model.add(tf.keras.layers.Dense(64, activation = 'relu'))

model.add(tf.keras.layers.Dense(64, activation = 'relu'))

model.add(tf.keras.layers.Dense(1))
model.summary()
model.compile(loss = tf.keras.losses.mse, #실제값과 예측값의 차이를 수치화하는 함수

             optimizer = tf.keras.optimizers.Adam(lr = 0.0001), #손실 함수의 값을 줄여나가면서 학습하는 방법은 어떤 옵티마이저를 사용하느냐에 따라 달라짐. lr(Learning Rate) 학습율 

             metrics = ['mse'])
history = model.fit(X_train_norm, Y_train_norm, batch_size = 300, epochs = 30, validation_split = 0.2) #batch_size=300
pd.DataFrame(history.history)
pd.DataFrame(history.history).reset_index()
pd.DataFrame(history.history).reset_index().plot('index','loss')
pd.DataFrame(history.history).reset_index().plot('index','val_loss')
Y_train_norm
submission['trip_duration'] = model.predict(X_test_norm) * Y_train.std() + Y_train.mean()
submission['trip_duration']
submission.to_csv('NYTaxi_NeuralNetwork_batchsize300.csv', index=False)