import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

import matplotlib.pyplot as plt

plt.style.use('seaborn-darkgrid')

import seaborn as sns



from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization

from keras.utils import np_utils

pulsar = pd.read_csv('../input/predicting-a-pulsar-star/pulsar_stars.csv',  dtype={"target_class": "category"})

pulsar = pulsar.sample(frac=1)

pulsar = pulsar.rename(columns={' Mean of the integrated profile':"mean_profile",

       ' Standard deviation of the integrated profile':"std_profile",

       ' Excess kurtosis of the integrated profile':"kurtosis_profile",

       ' Skewness of the integrated profile':"skewness_profile", 

       ' Mean of the DM-SNR curve':"mean_dmsnr_curve",

       ' Standard deviation of the DM-SNR curve':"std_dmsnr_curve",

       ' Excess kurtosis of the DM-SNR curve':"kurtosis_dmsnr_curve",

       ' Skewness of the DM-SNR curve':"skewness_dmsnr_curve",

       })
pulsar.shape
pulsar.groupby(['target_class'])[['target_class']].count()
pulsar[:5]
pulsarYes = pulsar[pulsar.target_class == "1"]

pulsarNo = pulsar[pulsar.target_class == "0"]

pulsarYes.shape, pulsarNo.shape
fig1 = plt.figure(1, figsize=(10,5))

sns.distplot(pulsarYes['skewness_profile']).set_title("pulsar star")



fig2 = plt.figure(2, figsize=(10,5))

sns.distplot(pulsarNo['skewness_profile']).set_title("non pulsar star")
from sklearn.model_selection import train_test_split



pulsarYesXTrain = pulsarYes.drop(columns=['target_class'])

pulsarNoXTrain = pulsarNo.drop(columns=['target_class'])



pulsarYesYTrain = pulsarYes[['target_class']]

pulsarNoYTrain = pulsarNo[['target_class']]



X_train_Yes, X_test_Yes, y_train_Yes, y_test_Yes = train_test_split(pulsarYesXTrain, pulsarYesYTrain, test_size=0.3)

X_train_No, X_test_No, y_train_No, y_test_No = train_test_split(pulsarNoXTrain, pulsarNoYTrain, test_size=0.33)



X_train = pd.concat([X_train_Yes, X_train_No])

X_test = pd.concat([X_test_Yes, X_test_No])



y_train = pd.concat([y_train_Yes, y_train_No])

y_test = pd.concat([y_test_Yes, y_test_No])

print(pulsarYesXTrain.shape,pulsarNoXTrain.shape,pulsarYesYTrain.shape,pulsarNoYTrain.shape)



print(X_train.shape, y_train.shape)

print(X_test.shape, y_test.shape)



model = Sequential()

model.add(Dense(4, activation='relu', input_dim=8))

model.add(BatchNormalization())

model.add(Dropout(0.25))

model.add(Dense(6, activation='relu'))

model.add(BatchNormalization())

model.add(Dense(6, activation='relu'))

model.add(Dropout(0.25))

model.add(Dense(4, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.compile('adam', loss='mean_squared_error')



model.fit(X_train, y_train, epochs=4)

predProba = model.predict(X_test)

pred = (predProba >0.5).astype(np.int)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test.astype(np.int), pred)

cm


ax= plt.subplot()

sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells



# labels, title and ticks

ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 

ax.set_title('Confusion Matrix'); 

ax.xaxis.set_ticklabels(['not pulsar', 'pulsar']); ax.yaxis.set_ticklabels(['not pulsar', 'pulsar']);