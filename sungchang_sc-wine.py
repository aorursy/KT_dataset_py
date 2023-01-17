from keras import regularizers

from keras.layers import Dense, Input, Dropout

from keras.models import Sequential

from keras.utils import np_utils

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# importing data

reddata = pd.read_csv('/kaggle/input/wine-quality/wineQualityReds.csv')

reddata["type"] = "red"

whitedata = pd.read_csv('/kaggle/input/wine-quality/wineQualityWhites.csv')

whitedata["type"] = "white"

alldata = pd.merge(reddata, whitedata, how = "outer")



from sklearn.preprocessing import LabelEncoder, OneHotEncoder

le = LabelEncoder()

alldata['type'] = le.fit_transform(alldata['type'])
alldata.head(10)
alldata.describe()
correlation_matrix = alldata.corr()

fig = plt.figure(figsize=(12,9))

sns.heatmap(correlation_matrix,vmax=0.4, vmin=-0.4,linewidths=1, annot=True)

plt.show()
sns.pairplot(alldata,kind="reg")

plt.show()
quality = alldata['quality']

alldata = alldata.drop(['quality'], axis = 1)
#X = alldata[['alcohol', 'volatile.acidity', 'chlorides','density','type']].values

X = alldata[['fixed.acidity', 'volatile.acidity', 'citric.acid', 'residual.sugar', 'chlorides', 'free.sulfur.dioxide', 'total.sulfur.dioxide', 'density', 'pH', 'sulphates', 'alcohol','type']].values

#X = alldata.iloc[: , 1:-2].values

Y = quality.values

Y = Y.reshape(-1,1)

Y = np_utils.to_categorical(Y)
from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_norm = sc.fit_transform(X)

#Y_norm = sc.fit_transform(Y)
# Splitting the dataset into the Training set and Test set for evaluation 트레인 25% 분리

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_norm,Y,test_size=0.25, random_state = 42)
# NN의 정의

# SGD 보다 아담이 잘됨

NNinput = X_train.shape[1]

NNoutput = y_train.shape[1]

act = 'relu'

opt = 'Adam'

los = 'categorical_crossentropy'



model = Sequential()

model.add(Dense(128, activation = act, input_shape = [NNinput,], activity_regularizer=regularizers.l2()))

model.add(Dense(128, activation = act))

model.add(Dense(128, activation = act))

model.add(Dense(256, activation = act))

model.add(Dropout(0.3))

model.add(Dense(NNoutput, activation = 'softmax'))

model.compile(optimizer = opt, loss = los, metrics = ['acc'])
epoch = 10

history = model.fit(X_train, y_train, epochs = epoch, batch_size = 20, verbose = 2, validation_data = [X_test, y_test])
pred = model.predict(X_test)
pred_norm = pred.argmax(axis=1)

y_test_M = y_test.argmax(axis=1)

y_pred_M = np.reshape(pred_norm, (1, X_test.shape[0]))[0]

totalPer = np.mean(1 - abs(1 - (y_pred_M / y_test_M)))

totalsub = np.mean(abs(y_pred_M - y_test_M))

print(totalPer)

print(totalsub)
y_pred_M = y_pred_M.reshape(len(y_pred_M), 1)

y_test_M = y_test_M.reshape(len(y_test_M), 1)



total = np.concatenate((y_test_M, y_pred_M), axis=1)

print(total[:100])
######################## DTC

from sklearn.tree import DecisionTreeClassifier

DTCmodel=DecisionTreeClassifier(max_depth=64, random_state=18)

DTCmodel.fit(X_train, y_train)

DTC_predN=DTCmodel.predict(X_test)

from sklearn.metrics import confusion_matrix

CM_test = np.argmax(y_test, axis=1)

CM_pred = np.argmax(DTC_predN, axis=1)

#cm = confusion_matrix(CM_test,CM_pred)

#print(cm)
CM_test = CM_test.reshape(CM_test.shape[0], 1)

CM_pred = CM_pred.reshape(CM_pred.shape[0], 1)

totalPer = np.mean(1 - abs(1 - (CM_pred / CM_test)))

totalsub = np.mean(abs(CM_pred - CM_test))

print(totalPer)

print(totalsub)