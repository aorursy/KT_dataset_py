# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
%matplotlib inline
train_data = pd.read_csv('../input/regularization-techniques/train.csv')
train_data.head(20)
test_data = pd.read_csv('../input/regularization-techniques/test.csv')
test_data.head(20)
tempvar = train_data.drop('label', axis=1)
image = tempvar.loc[0].to_numpy()
image = image.copy()
image.resize(28, 28)
imageplot = plt.imshow(image)
def plotColDistrbn(dfm, nGraph, nGraphsRow):
    numunique = dfm.nunique()
    dfm = dfm[[col for col in dfm if numunique[col] > 1 and numunique[col] < 50]] 
    nRow, nCol = dfm.shape
    colName = list(dfm)
    nGraphRow = (nCol + nGraphsRow - 1) / nGraphsRow
    plt.figure(num = None, figsize = (6 * nGraphsRow, 8 * nGraphRow), dpi = 80, facecolor = 'brown', edgecolor = 'b')
    for i in range(min(nCol, nGraph)):
        plt.subplot(nGraphRow, nGraphsRow, i + 1)
        coldfm = dfm.iloc[:, i]
        if (not np.issubdtype(type(coldfm.iloc[0]), np.number)):
            vCount = coldfm.value_count()
            vCount.plot.bar()
        else:
            coldfm.hist()
        plt.ylabel('count')
        plt.xticks(rotation = 90)
        plt.title(f'{colName[i]} (col {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()
plotColDistrbn(train_data, 10, 5)
plotColDistrbn(test_data, 10, 5)
def plotCorrMatrix(dfm, graphwidth):
    dfm = dfm[[col for col in dfm if dfm[col].nunique() > 1]] 
    if dfm.shape[1] < 2:
        print(f'There are no correlation plots to be shown, The number ofconstant col ({dfm.shape[1]}) is less than 2')
        return
    corr = dfm.corr()
    plt.figure(num=None, figsize=(graphwidth, graphwidth), dpi=80, facecolor='black', edgecolor='b')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    plt.show()
plotCorrMatrix(train_data, 25)
plotCorrMatrix(test_data, 25)
x_train = train_data.drop('label', axis=1).to_numpy()
x_train = x_train.reshape(-1, 784).astype('float32')
y_train = train_data['label'].to_numpy()
x_test = test_data.drop('label', axis=1).to_numpy()
y_test = test_data['label'].to_numpy()
input_num_of_units = 784
hidden1_num_of_units = 500
hidden2_num_of_units = 500
output_num_of_units = 1
epochs = 5
batch_size = 128
model = Sequential([
 Dense(units=hidden1_num_of_units, input_dim=input_num_of_units, activation='relu'),
 Dense(units=hidden2_num_of_units, activation='relu'),

Dense(units=output_num_of_units, activation='softmax'),
 ])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
trained_model_5d = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))
from keras import regularizers
model = Sequential([
 Dense(units=hidden1_num_of_units, input_dim=input_num_of_units, activation='relu',
 kernel_regularizer=regularizers.l2(0.0001)),
 Dense(units=hidden2_num_of_units, input_dim=hidden1_num_of_units, activation='relu',
 kernel_regularizer=regularizers.l2(0.0001)),

Dense(units=output_num_of_units, input_dim=hidden2_num_of_units, activation='softmax'),
 ])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
trained_model_5d = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))
model = Sequential([
 Dense(units=hidden1_num_of_units, input_dim=input_num_of_units, activation='relu',
 kernel_regularizer=regularizers.l1(0.0001)),
 Dense(units=hidden2_num_of_units, input_dim=hidden1_num_of_units, activation='relu',
 kernel_regularizer=regularizers.l1(0.0001)),
 
Dense(units=output_num_of_units, input_dim=hidden2_num_of_units, activation='softmax'),
 ])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
trained_model_5d = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))
from keras.layers.core import Dropout
model = Sequential([
 Dense(units=hidden1_num_of_units, input_dim=input_num_of_units, activation='relu'),
 Dropout(0.25),
 Dense(units=hidden2_num_of_units, input_dim=hidden1_num_of_units, activation='relu'),
 Dropout(0.25),

Dense(units=output_num_of_units, input_dim=hidden2_num_of_units, activation='softmax'),
 ])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
trained_model_5d = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))
from keras.preprocessing.image import ImageDataGenerator
datagenerator = ImageDataGenerator(zca_whitening=True)
x_train = np.stack(train_data.drop('label', axis=1).to_numpy())
X_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
X_train = X_train.astype('float32')
datagenerator.fit(X_train)
y_train = train_data['label'].to_numpy()
x_test = test_data.drop('label', axis=1).to_numpy()
y_test = test_data['label'].to_numpy()
x_train=np.reshape(x_train,(x_train.shape[0],-1))/255
x_test=np.reshape(x_test,(x_test.shape[0],-1))/255
from keras.layers.core import Dropout
model = Sequential([
 Dense(units=hidden1_num_of_units, input_dim=input_num_of_units, activation='relu'),
 Dropout(0.25),
 Dense(units=hidden2_num_of_units, input_dim=hidden1_num_of_units, activation='relu'),
 Dropout(0.25),
Dense(units=output_num_of_units, input_dim=hidden2_num_of_units, activation='softmax'),
 ])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
trained_model_5d = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))
from keras.callbacks import EarlyStopping
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
trained_model_5d = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test)
 , callbacks = [EarlyStopping(monitor='val_acc', patience=2)])