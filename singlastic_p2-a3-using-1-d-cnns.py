import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import random
import copy
RANDOM_SEED = 7
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

from sklearn.metrics import mean_squared_error
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dropout, Flatten
import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.models import load_model
from keras import regularizers
from keras import losses
from keras import optimizers

def read_signal(signal_number):
    x = pd.read_csv('../input/s'+signal_number+'.csv').T.values[::, ::].astype(float)
    return x

def prepare_data():
    y = pd.read_csv('../input/GT.csv')
    print("read data")
    y = y.T.values[::, ::].astype(float)
    x = [0]*30
    for i in range(1,31):
        z = read_signal(str(i))
        x[i-1] = z[0]

    return [x,y]
    
def prepare_val_data(x,y):    
    np.random.seed(RANDOM_SEED)
    val_index = np.random.randint(y.shape[0])
    x_val = x.pop(val_index)
    y_val = y[val_index]
    x_train = x
    y_train = np.delete(y, val_index, axis = 0)
    return [x_train, y_train, x_val, y_val]
def score_time(x_train, y_train):
    variance = 50 #hyperparameter
    scores = copy.deepcopy(x_train)
    for i in range(len(x_train)):
        timestamps = np.unique(y_train[i])
        for j in range(x_train[i].size):
            min = np.min(np.abs(timestamps-j))
            scores[i][j] = np.exp(-1*((min/variance)**2))
    return scores
def prepare_2D_window_vector(x_train, y_train, window_size = 50):
    scores = score_time(x_train,y_train)
    print("scores ready")
    x_input = []
    y_input = []
    
    prev_index = 0
    x_input = np.array(copy.deepcopy(x_train[0][0:window_size]))
    y_input = np.mean(scores[0][0:window_size])
    
    for i in range(len(x_train)):
        x_input = np.vstack([x_input, x_train[i][0:window_size]])
        y_input = np.append(y_input, np.mean(scores[i][0:window_size]))

        prev_index = 0
        while prev_index < (x_train[i].size - window_size-10):
            stride = 10
            if(np.min(np.abs(prev_index-y_train[i]))<30):
                stride = 2
            
            x_input = np.vstack([x_input, x_train[i][prev_index + stride: prev_index + stride + window_size]])
            y_input = np.append(y_input, np.mean(scores[i][prev_index + stride: prev_index + stride + window_size]))
            prev_index = prev_index + stride
            
    return [x_input, y_input]

def make_test_windows(x_val, window_size = 50):
    prev_index = 0
    test_windows = np.array(x_val[0:window_size])
    
    while prev_index < (x_val.size - window_size - 3):
        stride = 2
        test_windows = np.vstack([test_windows, x_val[prev_index + stride:prev_index + stride + window_size]])
        prev_index = prev_index + stride
    
    return test_windows

def make_predictions_max_scores(test_scores, window_size = 50):
    idx = np.argsort(test_scores)
    n = window_size/2
    a_pred = [2*idx[0]+n]
    for i in range(1, len(test_scores)):
        flag = False
        for j in range(len(a_pred)):
            if(np.abs(a_pred[j]-2*idx[i]-n) < 50):
                flag = True
                break
        if(flag == False):
            a_pred.append(2*idx[i]+n)
        if(len(a_pred) == 20):
            break
    return a_pred

def make_predictions_less_misses(test_scores, window_size=50, threshold = 0.75):
    a_pred = []
    n = window_size/2
    for i in range(len(test_scores)):
        if(test_scores[i] > threshold):
            a_pred.append(2*i+n)
    a_final = []
    i = 0
    while(i < len(a_pred)):
        j = a_pred[i]
        max_score = 0
        max_idx = 0
        while(i < len(a_pred) and np.abs(a_pred[i]-j) <= 50):
            k = int((a_pred[i]-n)/2)
            if(max_score <= test_scores[k]):
                max_score = test_scores[k]
                max_idx = i
            i += 1
        a_final.append(a_pred[max_idx])
    return a_final

def accuracy(a_pred, a):
    misses = 0
    n = len(a_pred)
    m = len(a)
    false_alarms = 0
    correct_predictions = 0
    counted = []
    std = 0
    for i in range(len(a)):
        count = 0
        for j in range(len(a_pred)):
            if(np.abs(a_pred[j]-a[i]) <= 50 and j not in counted):
                std += (a_pred[j]-a[i])**2
                counted.append(j)
                count += 1
        if(count == 0):
            misses += 1
        elif(count == 1):
#             print("i= " + str(i))
            correct_predictions += 1
        else:
            correct_predictions += 1
            false_alarms += count-1
    false_alarms += n-len(counted)
#     print(str(n) + " " + str(correct_predictions))
    if(n == 0):
        n = 1
    return correct_predictions/n, misses/m, false_alarms/n, std**0.5/n
[x_train,y_train] = prepare_data()

[x_train, y_train, x_val, y_val] = prepare_val_data(x_train,y_train)
[x_input, y_input] = prepare_2D_window_vector(x_train, y_train)
print("Prepared Data")


K.clear_session()

model_m = Sequential()
# model_m.add(Conv1D(filters = 32, kernel_size = 4, activation='relu', input_shape= (50,1)))
# model_m.add(Conv1D(filters = 32, kernel_size = 4, activation='relu'))
# model_m.add(MaxPooling1D(2))
# model_m.add(Conv1D(filters = 50, kernel_size = 4, activation='relu'))
# model_m.add(Conv1D(filters = 50, kernel_size = 4, activation='relu'))
# model_m.add(GlobalAveragePooling1D())
# model_m.add(Dropout(0.5))
# model_m.add(Flatten())
model_m.add(Dense(100, activation = 'relu', kernel_initializer = 'normal'))
model_m.add(Dense(1, kernel_initializer = 'normal', activation = 'relu'))
model_m.compile(loss='mean_squared_error', optimizer='Adam',)

EPOCHS = 50
BATCH_SIZE = 100
model_m.fit(x_input, y_input, batch_size = BATCH_SIZE, epochs = EPOCHS, shuffle = True)
test_windows = make_test_windows(x_val)
test_scores = model_m.predict(test_windows)
test_scores = np.reshape(test_scores, test_scores.shape[0])
a_pred = make_predictions_less_misses(test_scores, 50, 0.75)
[acc, misses, false_alarms, std] = accuracy(a_pred, np.unique(y_val))
print(a_pred)
print(np.unique(y_val))
print(acc)
print(misses)
print(false_alarms)
print(std)
acc_arr = []
miss_arr = []
fa_arr = []
t_arr = []
t = 0.0
while(t<=1):
    a_pred = make_predictions_less_misses(test_scores, 50, t)
    [acc, misses, false_alarms] = accuracy(a_pred, np.unique(y_val))
    t_arr.append(t)
    t += 0.02
    acc_arr.append(acc)
    miss_arr.append(misses)
    fa_arr.append(false_alarms)

plt.plot(t_arr,acc_arr, label = 'accuracy')
plt.plot(t_arr,miss_arr, label = 'misses')
plt.plot(t_arr,fa_arr, label = 'false_alarms')
plt.xlabel('Threshold')
plt.ylabel('Value')
plt.title('Accuracy, misses and false alarms vs threshold')
plt.axis('on')
plt.legend()
a = []
for i in range(29):
    test_windows = make_test_windows(x_train[i])
    test_scores = model_m.predict(test_windows)
    test_scores = np.reshape(test_scores, test_scores.shape[0])
    a.append(make_predictions_less_misses(test_scores, 50, 0.75))
a
for j in range(29):
    plt.figure(figsize=(20,5))
    ax = plt.plot(x_train[j].T)
    plt.xlim(xmin=0)
    xcoords = y_train[j]
    for xc in xcoords:
        plt.axvline(x=xc,color='red')
    xcoords = a[j]
    for xc in xcoords:
        plt.axvline(x=xc,color='black', linestyle="--")

