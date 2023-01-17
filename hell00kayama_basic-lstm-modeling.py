from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# detect and init the TPU
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)

# instantiate a distribution strategy
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

df = pd.read_csv('../input/ntt-data-global-ai-challenge-06-2020/Crude_oil_trend_From1986-01-02_To2020-06-15.csv')
df.head()
def make_dataset(low_data, maxlen=25):
    data, target = [], []
    
    for i in range(len(low_data)-maxlen):
        data.append(low_data[i:i + maxlen])
        target.append(low_data[i + maxlen])
        
    re_data = np.array(data).reshape(len(data), maxlen, 1)
    re_target = np.array(target).reshape(len(data),1)
    
    return re_data, re_target

orgi_data = df[(df.Date>='1999-10-20') & (df.Date<'2020-04-22')].iloc[:,1].values
print(len(orgi_data))
print(orgi_data[5144])
g, h = make_dataset(orgi_data)
print(g[5119],h[5119])
length_of_sequence = g.shape[1] 
in_out_neurons = 1
n_hidden = 300
with tpu_strategy.scope():
    model = Sequential()
    model.add(LSTM(n_hidden, batch_input_shape=(None, length_of_sequence, in_out_neurons), return_sequences=False))
    model.add(Dense(in_out_neurons))
    model.add(Activation("linear"))
    optimizer = Adam(lr=0.001)
    model.compile(loss="mean_squared_error", optimizer=optimizer)

early_stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=20)
model.fit(g, h,
          batch_size=300,
          epochs=100,
          validation_split=0.1,
          callbacks=[early_stopping]
          )
predicted = model.predict(g)
plt.figure()
plt.plot(range(25,len(predicted)+25),predicted, color="r", label="predict_data")
plt.plot(range(0, len(orgi_data)), orgi_data, color="b", label="row_data")
plt.legend()
plt.show()
future_test = g[5119].T
time_length = future_test.shape[1]
test_data = np.empty((0))
future_result = np.empty((0))

for step in range(39):
    test_data= np.reshape(future_test, (1, time_length, 1))
    batch_predict = model.predict(test_data)
    future_test = np.delete(future_test, 0)
    future_test = np.append(future_test, batch_predict[0])
    future_result = np.append(future_result, batch_predict[0])
    
print(future_result.shape)
print(future_result)
fig = plt.figure(figsize=(10,5),dpi=200)
sns.lineplot(
    color="#086039",
    data=orgi_data[5069:],
    label="Raw Data",
    marker="o"
)

sns.lineplot(
    color="#f44262",
    x=np.arange(25, len(predicted[5069:])+25),
    y=predicted[5069:].reshape(-1),
    label="Predicted Training Data",
    marker="o"
)

sns.lineplot(
    color="#a2fc23",
    x = np.arange(0+len(orgi_data[5069:]), len(future_result)+len(orgi_data[5069:])),
    y= future_result.reshape(-1),
    label="Predicted Future Data",
    marker="o"
)
ground_truth = df[(df.Date>='2020-04-29') & (df.Date<='2020-06-15')].iloc[:,1].values
ground_truth
future_result[6:]
np.sqrt(mean_squared_error(ground_truth, future_result[6:]))
