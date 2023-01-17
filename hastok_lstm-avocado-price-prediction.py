import numpy as np #For linear algebra maths
import pandas as pd #For data manipulation
from keras.models import Sequential #For DL
from keras.layers import LSTM, Dense, Dropout #For DL
import matplotlib.pyplot as plt #For plotting
hist_len = 10
data = pd.read_csv("../input/avocado.csv")
data.head()
data = data.drop(['Unnamed: 0','Date','Total Volume','4046','4225','4770','Total Bags','Small Bags','Large Bags','XLarge Bags','type','year','region'],1)
data = data.T
data = data.values
data = data[0]
scale_min = min(data)
scale_range = max(data) - scale_min
data = (data-scale_min)/(scale_range)
def make_feed_dicts(data,hist_len):
    xs,ys = [],[]
    for i in range(len(data)-hist_len-1):
        ys.append(data[i+hist_len])
        xs.append(data[i:i+hist_len])
    j = int(len(data)*0.9)
    return np.array(xs[:j]),np.array(xs[j:]),np.array(ys[:j]),np.array(ys[j:])
x_train, x_test, y_train, y_test = make_feed_dicts(data,hist_len)
x_test.shape
x_train = x_train.reshape((x_train.shape[0],x_train.shape[1],1))
x_test = x_test.reshape((x_test.shape[0],x_test.shape[1],1))
x_test.shape
model = Sequential()
model.add(LSTM(256,input_shape=(hist_len,1)))
model.add(Dense(5))
#model.add(Dropout(0.1))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.summary()
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test,y_test),shuffle=False)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
predicted_x = model.predict(x_test[:250])
plt.plot(predicted_x*scale_range+scale_min)
plt.plot(y_test[:250].reshape(-1,1)*scale_range+scale_min)
step = np.array([x_test[0]])



def take_step(step):
    next_step = model.predict(step)
    next_step = next_step.reshape(1,1,1)
    next_step = np.concatenate(([step[0][1:]],next_step),axis=1)
    return next_step, next_step[0][0][0]
    
    
trendline = []
for _ in range(100):
    step, value = take_step(step)
    trendline.append(value)
    
plt.plot(trendline)
plt.plot(y_test[7:100])