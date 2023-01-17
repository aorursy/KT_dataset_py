# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import pandas as pd

import numpy as np # linear algebra

from numpy import mean

from numpy import std

from numpy import dstack

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Flatten

from keras.layers import Dropout

from keras.layers import LSTM

from keras.utils import to_categorical

from matplotlib import pyplot

import matplotlib.pyplot as plt

from scipy import stats

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

DATA_PATH = '/kaggle/input/wisdmdata/WISDM_ar_v1.1_raw.txt'

COLUMN_NAMES = [

    'user',

    'activity',

    'timestamp',

    'x-axis',

    'y-axis',

    'z-axis'

]



LABELS = [

    'Downstairs',

    'Jogging',

    'Sitting',

    'Standing',

    'Upstairs',

    'Walking'

]
RANDOM_SEED = 13

N_TIME_STEPS = 50

N_FEATURES = 3

step = 10

segments = []

labels = []

N_CLASSES = 6

N_HIDDEN_UNITS = 64

N_EPOCHS = 20

BATCH_SIZE = 1024

LEARNING_RATE = 0.0025

L2_LOSS = 0.0015

def readData(DATA_PATH,COLUMN_NAMES,LABELS):

    data = pd.read_csv(DATA_PATH, header=None, names=COLUMN_NAMES)

    data['z-axis'].replace({';': ''}, regex=True, inplace=True)

    data['z-axis']=pd.to_numeric(data['z-axis'],errors='coerce')

    data = data.dropna() 

    return data

def plotActivity(data,activity,rowsSize,subPlotObj,subPlotTitle):

    activityX= data[data['activity'] == activity][['x-axis']][:rowsSize]

    activityY= data[data['activity'] == activity][['y-axis']][:rowsSize]

    activityZ= data[data['activity'] == activity][['z-axis']][:rowsSize]

    subPlotObj.plot(activityX, label='X')

    subPlotObj.plot(activityY,'--', label='Y')

    subPlotObj.plot(activityZ, label='Z')

    subPlotObj.set_title(subPlotTitle)

    subPlotObj.legend()

def plot_correlation(df,subPlotObj,subPlotTitle):   

   sns.heatmap(df.corr(),cmap="YlGnBu",cbar=True,ax=subPlotObj)

   tl = subPlotObj.get_xticklabels()

   subPlotObj.set_xticklabels(tl, rotation=90)

   tly = subPlotObj.get_yticklabels()

   subPlotObj.set_yticklabels(tly, rotation=0)

   subPlotObj.set_title(subPlotTitle)

def create_mixed_Df(activity1, activity2,subPlotObj):

    activity1X=projectData[projectData['activity'] == activity1]['x-axis'][:]

    activity1Y= projectData[projectData['activity'] == activity1]['y-axis'][:]

    activity1Z=projectData[projectData['activity'] == activity1]['z-axis'][:]

    activity2X=projectData[projectData['activity'] == activity2]['x-axis'][:]

    activity2Y= projectData[projectData['activity'] == activity2]['y-axis'][:]

    activity2Z=projectData[projectData['activity'] == activity2]['z-axis'][:]

    list_of_tuples = list(zip(activity1X,activity1Y,activity1Z, activity2X,activity2Y,activity2Z))      

    df = pd.DataFrame(list_of_tuples, columns = [activity1+'X', activity1+'Y',activity1+'Z',activity2+'X',activity2+'Y',activity2+'Z'])     
projectData=readData(DATA_PATH,COLUMN_NAMES,LABELS)
#create segements each segement contains 50 (N_TIME_STEPS) activity records and interval 10 (step) stepsize

for i in range(0, len(projectData) - N_TIME_STEPS, step):

    xs = projectData['x-axis'].values[i: i + N_TIME_STEPS]

    ys = projectData['y-axis'].values[i: i + N_TIME_STEPS]

    zs = projectData['z-axis'].values[i: i + N_TIME_STEPS]

    label = stats.mode(projectData['activity'][i: i + N_TIME_STEPS])[0][0]

    segments.append([xs, ys, zs])

    labels.append(label)



#reshape the segments which is (list of arrays) to one list

reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, N_TIME_STEPS, N_FEATURES)

labels = np.asarray(pd.get_dummies(labels), dtype = np.float32)
X_train, X_test, y_train, y_test = train_test_split(

        reshaped_segments, labels, test_size=0.2, random_state=RANDOM_SEED)
X_train.shape[1]
epochs, batch_size =  25, 1024

model = Sequential()

model.add(LSTM(100, input_shape=(X_train.shape[1],X_train.shape[2])))

model.add(Dropout(0.4))

model.add(Dense(100, activation='relu'))

model.add(Dense(y_train.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=epochs, validation_split=0.25,batch_size=batch_size,verbose=1)
_, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)

accuracy
from keras.models import load_model

model.save('harmodel.h5')  # creates a HDF5 file 'my_model.h5'

# Save the model architecture

with open('model_architecture.json', 'w') as f:

    f.write(model.to_json())
plt.figure(figsize=(12, 8))

plt.plot(np.array(history.history['loss']), "r--", label="Train loss")

plt.plot(np.array(history.history['accuracy']), "g--", label="Train accuracy")

plt.plot(np.array(history.history['val_loss']), "r-", label="Validation loss")

plt.plot(np.array(history.history['val_accuracy']), "g-", label="Validation accuracy")

plt.title("Training session's progress over iterations")

plt.legend(loc='upper right', shadow=True)

plt.ylabel('Training Progress (Loss or Accuracy values)')

plt.xlabel('Training Epoch')

plt.ylim(0)

plt.show()
LABELS = ['Downstairs', 'Jogging', 'Sitting', 'Standing', 'Upstairs', 'Walking']

y_pred = model.predict(X_test)

matrix = confusion_matrix(y_test.argmax(axis=1),y_pred.argmax(axis=1))
plt.figure(figsize=(16, 14))

sns.heatmap(matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");

plt.title("Confusion matrix")

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show();