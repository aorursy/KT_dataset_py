# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Reading the Dataset

df = pd.read_csv("../input/Sunspots.csv")
# taking just the Last 100 years for training the Model

# Instead, the last 2 years it will be the Testing data.

lastYearsWindow = 13

last100YearsWindow = 1201

df100Years = df[df.index > ( df['Date'].count() - last100YearsWindow )  ]

dfLastYears = df[ df.index > ( df['Date'].count() - lastYearsWindow)]

df100Years = df100Years[:-lastYearsWindow+1]

# dfLast100Years

# dfLastYears.dtypes

# npYears = dfLastYears.to_numpy()

# npYears = dfLastYears.values
# --------------------------------------------------

# Creating the Year and Month data input from the DATE field in the Dataframe. 

# Dropping the not used Columns

# --------------------------------------------------

dfLastYears.loc[:,'Year']  = dfLastYears['Date'].map(lambda x: x.split('-')[0])

dfLastYears.loc[:,'Month'] = dfLastYears['Date'].map(lambda x: x.split('-')[1])

df100Years.loc[:,'Year']  = df100Years['Date'].map(lambda x: x.split('-')[0])

df100Years.loc[:,'Month'] = df100Years['Date'].map(lambda x: x.split('-')[1])

# Drop DATE

dfLastYears = dfLastYears.drop(["Date", "Unnamed: 0"], axis=1)

# dfLastYears = dfLastYears[: 2:]

df100Years  = df100Years.drop(["Date","Unnamed: 0"], axis=1)

# df100Years  = df100Years[: 2:]







# dfLastYears[['Month','Year','Monthly Mean Total Sunspot Number']]

# Check data

df100Years.tail(5)

# 

# df100Years.values.shape

#

# 

# df100Years['Monthly Mean Total Sunspot Number'].describe()
dfLastYears[:]
# df100Years.shape[:-1] + ( df100Years.shape[-1], 3)

# trainingInput = df100Years['Monthly Mean Total Sunspot Number'].values.reshape(1, df100Years['Monthly Mean Total Sunspot Number'].count(),10)

# --------------------------------------------------

# Create a Sequence of Inputs and results for the them.

# the Inputs could be sampled with the size parameter.

def to_sequences(seq_size, data_obs):

    x = []

    y = []

    # for each window 

    for i in range(len(data_obs) - seq_size - 1):

        window = data_obs[i:(i+seq_size)]

        after_window = data_obs[(i+seq_size)]

        window = [ [x] for x in window ]

        x.append(window)

        y.append(after_window)

    return np.array(x), np.array(y)



# Applying the Function in order to build the sequences as inputs

time_window = 5

x_train, y_train = to_sequences(time_window, df100Years['Monthly Mean Total Sunspot Number'].tolist())

x_test, y_test   = to_sequences(time_window, dfLastYears['Monthly Mean Total Sunspot Number'].tolist())





# SHAPE

x_train.shape, x_test.shape

# first 3 rows

# x_train[:3]

# 

trainMax= x_train.max()

resultTrainMax= y_train.max()

testMax= x_test.max()

resultTestMax=y_test.max()



# Normalization to 1

x_train = x_train/trainMax

y_train = y_train/resultTrainMax

x_test  = x_test/testMax

y_test  = y_test/resultTestMax

x_test.shape
x_train.shape[1:]
# Keras Model 

from keras.models import Sequential

from keras.layers import Dense, LSTM, Dropout

from keras.optimizers import Adagrad, SGD, RMSprop

# number od EPOCHS on the train data set

epochs_value = 100

print('Config the Model...')

# automatic definition of the Shape using x_train.shape[1:]

my_model = Sequential()

my_model.add(LSTM(24, dropout=0.2, recurrent_dropout=0.2, activation='relu', input_shape=x_train.shape[1:], return_sequences=True ))

# my_model.add(Dropout(0.2))

my_model.add(LSTM(12, dropout=0.1, recurrent_dropout=0.1, activation='relu'))

# my_model.add(Dropout(0.2))

#  activation='linear'

my_model.add(Dense(1))

# loss='mean_squared_error'  'logcosh'

# optimizer='adam'

# opt = Adagrad(lr=0.001, epsilon=None, decay=0.0001)

# opt = SGD(lr=0.01, decay=1e-6, momentum=0.8)



# Based on Experiment, the RMSprop is the best Optimer - decay=1e-6

opt = RMSprop(lr=0.01, decay=0.0, rho=0.85)

my_model.compile(loss='mean_squared_error', optimizer=opt)



print('Training...')

history = my_model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=24,epochs=epochs_value)









import matplotlib.pyplot as plt



# Plot training & validation accuracy values



plt.plot(history.history['loss'])

plt.title('Model accuracy')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
from sklearn import metrics

prediction = my_model.predict(x_test)

score = np.sqrt(metrics.mean_squared_error(prediction, y_test))

print("Score (RMSE) : {}".format(score))

print("Not Normalized Score (RMSE) : {}".format(score*resultTestMax))
prediction*testMax
# print("X real values ")

# print(*(x_test*testMax), sep=", ")

print("Y real values ")

print(*(y_test*resultTestMax), sep=", ")

print("versus forecast") 

print(*(prediction*testMax), sep=", ")

print("Diff") 

diff = []

for x in range(len(y_test)):

    diff.append( (prediction[x]*testMax) - (y_test[x]*resultTestMax) )

print(*diff)