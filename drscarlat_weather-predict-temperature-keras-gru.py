# IMPORT MODULES
import sys
from os.path import join
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array

from keras import models, regularizers, layers, optimizers, losses, metrics
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils, to_categorical, Sequence
from keras.optimizers import RMSprop
from keras.models import load_model

import os
print(os.listdir("../input"))
#data_dir = '/input/Weather-'
#fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')
f = open('../input/weather-archive-jena/jena_climate_2009_2016.csv')
data = f.read()
f.close()
lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]
print(len(lines))
for i,j in enumerate(header):
    print(i,j)
# Convert lines list into Numpy array float_data

float_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i, :] = values
              
print(float_data.shape)
temp = float_data[:, 1] 
plt.plot(range(len(temp)), temp)
# Normalize data

mean = float_data[:200000].mean(axis=0)
float_data -= mean
std = float_data[:200000].std(axis=0)
float_data /= std

float_data
# PLOT ranges of float_data from VALIDATION range
Ulim = 202000
Llim = 200000
SamSize = Ulim - Llim

plt.plot(range(SamSize), temp[Llim:Ulim], 'r')
#plt.plot(range(SamSize), pBar[Llim:Ulim], 'g')
#plt.plot(range(SamSize), rh[Llim:Ulim], 'b')
def generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)
        samples = np.zeros((len(rows),lookback // step,data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets
lookback = 1440
step = 6
delay = 144
batch_size = 128

train_gen = generator(float_data,
                        lookback=lookback,
                        delay=delay,
                        min_index=0,
                        max_index=200000,
                        shuffle=True,
                        step=step)

val_gen = generator(float_data,
                        lookback=lookback,
                        delay=delay,
                        min_index=200001,
                        max_index=300000,
                        step=step)

test_gen = generator(float_data,
                        lookback=lookback,
                        delay=delay,
                        min_index=300001,
                        max_index=None,
                        step=step)

val_steps = (300000 - 200001 - lookback) // 128
test_steps = (len(float_data) - 300001 - lookback) // 128
# Sanity check - Baseline: The temp tomorrow = temp today

def evaluate_naive_method():
    batch_maes = []
    for step in range(val_steps):
        samples, targets = next(val_gen)
        preds = samples[:, -1, 1]
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)
    print('MAE ', round(np.mean(batch_maes),5))
    print('Celsius MAE ', round(np.mean(batch_maes)* std[1],2))
    
evaluate_naive_method()
maeC = 0.28974
Ulim = 202000
Llim = 200000
SampleSize = Ulim - Llim

plt.plot(range(SampleSize), temp[Llim:Ulim], 'r')
plt.plot(range(SampleSize), temp[Llim:Ulim]+maeC, 'b', linewidth=0.3)
plt.plot(range(SampleSize), temp[Llim:Ulim]-maeC, 'b', linewidth=0.3)
# SAVE or LOAD model (Keras - all batteries included: architecture, weights, optimizer, last status in training, etc.)
# YOU supply this model.h5 file from previous training session(s) - expected as a data source by Kaggle

# SAVE model
#model.save('modelWeatherV6.h5')
#print("modelWeatherV6.h5 was saved")

# LOAD model
#del model
model = load_model('../input/weather-v9/modelWeatherV10.h5')
print("modelWeatherV10.h5 was loaded")

model.summary()
model.compile(optimizer=RMSprop(), loss='mae')
print("model compiled")
# PREDICT on ALL test data

preds = model.predict_generator(test_gen, test_steps)
print (type(preds))
print(len(preds))
# Set the params for the charts
lenPreds = len(preds)
predTemp = preds[0:lenPreds,:]
print(predTemp.shape)

# CHART Predicted 
UlimPred = lenPreds
#UlimPred = 10000
LlimPred = 0
SampleSizePred = UlimPred - LlimPred
plt.plot(range(SampleSizePred), predTemp[LlimPred:UlimPred], 'b', linewidth=0.27)
# Chart actual Test data 
# 420551 - 300k = 120551 samples
# 120551 - 119040 preds = 1511 ... this is the starting point of the predictions on the test scale 300k-420551

UlimReal = 420551
#UlimReal = 311511
LlimReal = 301511
SampleSizeReal = UlimReal - LlimReal
plt.plot(range(SampleSizeReal), temp[LlimReal:UlimReal], 'r', linewidth=0.27)
# Superimposed: model prediction (blue) vs reality (red)

# Chart Test
UlimReal = 420551
#UlimReal = 311511
LlimReal = 301511
SampleSizeReal = UlimReal - LlimReal
plt.plot(range(SampleSizeReal), temp[LlimReal:UlimReal], 'r', linewidth=0.27)

# Chart Predicted 
UlimPred = lenPreds
#UlimPred = 10000
LlimPred = 0
SampleSizePred = UlimPred - LlimPred
plt.plot(range(SampleSizePred), predTemp[LlimPred:UlimPred], 'b', linewidth=0.27)