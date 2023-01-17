import numpy as np 
import pandas as pd 
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers import Dense,LSTM,Flatten,SimpleRNN,GRU
from keras.models import Sequential
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
jena_data = pd.read_csv('/kaggle/input/jena-climate-2009-2016/jena_climate_2009_2016.csv',index_col = None)
jena_data = jena_data.drop(['Date Time'],axis=1)
jena_data.head()
jena_data.columns
temp = jena_data.iloc[:,1]
h20c = jena_data.iloc[:,9]
fig, axes = plt.subplots(4,2, figsize = (15,15))

for i in range(4):
    axes[i][0].plot(temp[:144*365*(i+1)],label=f'{i+1} year Temperature T (degC)')
    axes[i][1].plot(h20c[:144*365*(i+1)],label=f'{i+1} year H20c mmol/mol',color = 'g') 
    axes[i][0].legend()
    axes[i][1].legend()
mean = np.mean(jena_data.values,axis=0)
std = np.std(jena_data.values,axis=0)
jena_data = (jena_data-mean)/std
# %%timeit
# jena_data.mean()
# %%timeit
# np.mean(jena_data)
def generator(normalized_data, lookback, 
              delay, min_index, max_index,
             shuffle = False, batch_size = 128,
             step=6):
    if max_index is None:
        max_index = len(normalized_data)-delay-1
    i= min_index+lookback
    rows = None
    while(1):
        if shuffle:
            rows = np.random.randint(min_index+lookback, max_index, size = batch_size)
        else:
            if i + batch_size >=max_index:
                i = min_index+batch_size
            rows = np.arange(i,min(i+batch_size, max_index))
            i += len(rows)
        
        samples = np.zeros((len(rows),lookback//step,normalized_data.shape[-1]))
        targets = np.zeros((len(rows),))
        
        for j,row in enumerate(rows):
            indices = range(rows[j]-lookback,rows[j],step)
            samples[j] = normalized_data[indices]
            targets[j] = normalized_data[rows[j]+delay][1]
        
        yield samples, targets
# training
lookback = 1440
step=6
delay = 144
batch_size = 128

train_gen = generator(np.array(jena_data),lookback=lookback,delay=delay,min_index=0,max_index=200000,shuffle=True,
                     step=step,batch_size=batch_size)

val_gen = generator(np.array(jena_data),lookback=lookback,delay=delay,min_index=200001,max_index=300000,
                     step=step,batch_size=batch_size)

test_gen = generator(np.array(jena_data),lookback=lookback,delay=delay,min_index=300001,max_index=None,
                     step=step,batch_size=batch_size)
val_steps = 300000-200001-lookback
test_steps = (len(np.array(jena_data))-lookback)
model = Sequential()

model.add(Flatten(input_shape=(lookback//step,jena_data.shape[-1])))
model.add(Dense(32,activation='relu'))
model.add(Dense(1,))
model.compile(optimizer = 'rmsprop', loss='mae')
history = model.fit_generator(train_gen,steps_per_epoch=500,epochs=4,validation_data = val_gen,validation_steps=500)
plt.plot(history.history['val_loss'],label = 'val_loss')
plt.plot(history.history['loss'], label='loss')
plt.legend()

# GRU
model = Sequential()

model.add(GRU(32,input_shape=(lookback//step,jena_data.shape[-1])))
model.add(Dense(32,activation='relu'))
model.add(Dense(1,))
model.compile(optimizer = 'rmsprop', loss='mae')
# Change epochs to 5-10 when you use
history = model.fit_generator(train_gen,steps_per_epoch=500,epochs=1,validation_data = val_gen,validation_steps=500)
model.predict_generator(test_gen,500)

tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
    
    # instantiate a distribution strategy
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
#Trying GPUs. Didn't work
model = None
with tpu_strategy.scope():
    model = Sequential()

    model.add(GRU(32,input_shape=(lookback//step,jena_data.shape[-1])))
    model.add(Dense(32,activation='relu'))
    model.add(Dense(1,))
    model.compile(optimizer = 'rmsprop', loss='mae')
history_tpu = model.fit_generator(train_gen,steps_per_epoch=500,epochs=5,validation_data = val_gen,validation_steps=2000)