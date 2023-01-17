import tensorflow as tf

import numpy as np

from sklearn.metrics import f1_score
data_path = '/kaggle/input/data-without-drift/'

train_data_file = data_path + 'train_clean.csv'

test_data_file = data_path + 'test_clean.csv'



def get_data(filename, train=True):

  

    if(train):

        with open(filename) as training_file:

            split_size = 10

            data = np.loadtxt(training_file, delimiter=',', skiprows=1)

            signal = data[:,1]

            channels = data[:,2]

            signal = np.array_split(signal, split_size)

            channels = np.array_split(channels, split_size)

            data = None

        return np.array(signal), np.array(channels)

    else:

       with open(filename) as training_file:

            split_size = 4

            data = np.loadtxt(training_file, delimiter=',', skiprows=1)

            signal = data[:,1]

            signal = np.array_split(signal, split_size)

            data = None

       return np.array(signal)



train_signal , train_channels = get_data(train_data_file)

test_signal = get_data(test_data_file, train=False)



test_model_signal = np.zeros((5,1000000))

test_model_channel = np.zeros((5,1000000))

test_model_signal[0][:500000] = train_signal[0].flatten()

test_model_signal[0][500000:] = train_signal[1].flatten()

test_model_signal[1][:500000] = train_signal[2].flatten()

test_model_signal[1][500000:] = train_signal[6].flatten()

test_model_signal[2][:500000] = train_signal[3].flatten()

test_model_signal[2][500000:] = train_signal[7].flatten()

test_model_signal[3][:500000] = train_signal[4].flatten()

test_model_signal[3][500000:] = train_signal[9].flatten()

test_model_signal[4][:500000] = train_signal[5].flatten()

test_model_signal[4][500000:] = train_signal[8].flatten()





test_model_channel[0][:500000] = train_channels[0].flatten()

test_model_channel[0][500000:] = train_channels[1].flatten()

test_model_channel[1][:500000] = train_channels[2].flatten()

test_model_channel[1][500000:] = train_channels[6].flatten()

test_model_channel[2][:500000] = train_channels[3].flatten()

test_model_channel[2][500000:] = train_channels[7].flatten()

test_model_channel[3][:500000] = train_channels[4].flatten()

test_model_channel[3][500000:] = train_channels[9].flatten()

test_model_channel[4][:500000] = train_channels[5].flatten()

test_model_channel[4][500000:] = train_channels[8].flatten()
from sklearn.svm import SVC

models = []



specs = [[1.2,1],[0.1,1],[0.5,1],[7,0.01],[10,0.1]]



for k in range (5):

    print("starting training model no: ", k)

    x = test_model_signal[k].flatten()

    y = test_model_channel[k].flatten()

    y = np.array(y).astype(int)

    x = np.expand_dims(np.array(x),-1)

    model = SVC(kernel = 'rbf', C=specs[k][0],gamma = specs[k][1])

    samples= 400000

    #trains by splitting into 10 batches for faster training

    for i in range(10):

        model.fit(x[i*samples//10:(i+1)*samples//10],y[i*samples//10:(i+1)*samples//10])

    y_pred = model.predict(x[400000:500000])

    y_true = y[400000:500000]

    print(f1_score(y_true, y_pred, average=None))

    print(f1_score(y_true, y_pred, average='macro'))

    models.append(model)
model_ref = [0,2,4,0,1,3,4,3,0,2,0,0,0,0,0,0,0,0,0,0]

y_pred_all = np.zeros((2000000))

for pec in range(20):

  print("starting prediction of test batch no: ", pec)

  x_test = test_signal.flatten()[pec*100000:(pec+1)*100000]

  x_test = np.expand_dims(np.array(x_test),-1)

  test_pred = models[model_ref[pec]].predict(x_test)

  y_pred_1 = np.array(test_pred).astype(int)

  y_pred_all[pec*100000:(pec+1)*100000] = y_pred_1



y_pred_all = np.array(y_pred_all).astype(int)
model_ref = [0,0,1,2,3,4]

y_valid = np.zeros((1000000))

y_pred = np.zeros((1000000))

for k in range(6):

  x = train_signal[k].flatten()

  y = train_channels[k].flatten()

  y = np.array(y).astype(int)

  x = np.expand_dims(np.array(x),-1)

  model = models[model_ref[k]]

  y_pred[k*100000:(k+1)*100000] = model.predict(x[400000:500000])

  y_valid[k*100000:(k+1)*100000]=y[400000:500000]



print(f1_score(y_valid, y_pred, average=None))

print(f1_score(y_valid, y_pred, average='macro'))
import pandas as pd

sub = pd.read_csv('/kaggle/input/liverpool-ion-switching/sample_submission.csv')

sub.iloc[:,1] = y_pred_all

sub.to_csv('submission.csv',index=False,float_format='%.4f')

print("saved the file")