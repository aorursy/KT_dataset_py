# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Sequential

from keras.layers import LSTM,Dense

from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib.pyplot as plt

import os

print(os.listdir("../input"))

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))





# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/pulsar_stars.csv')

df.head()
correlation = df.corr()

# display(correlation)

plt.figure(figsize=(14, 12))

heatmap = sns.heatmap(correlation, annot=True, linewidths=0, vmin=-1, cmap="RdBu_r")



df = df[[' Mean of the integrated profile',

       ' Standard deviation of the integrated profile',

       ' Excess kurtosis of the integrated profile',

       ' Skewness of the integrated profile', ' Mean of the DM-SNR curve',

       ' Standard deviation of the DM-SNR curve',

       ' Excess kurtosis of the DM-SNR curve', ' Skewness of the DM-SNR curve',

       'target_class']]

df0 = df.target_class.values.tolist()

print(len(df0))

i = 0

ok = 0

ko = 0

while i < len(df0):

    if df0[i] == 0:

        ok = ok+1

    else:

        ko = ko+1

    i += 1



print(ok/(ko+ok))

print(ko/(ko+ok))

        
import pandas as pd

import numpy as np

import pickle

import matplotlib.pyplot as plt

from scipy import stats

import tensorflow as tf

import seaborn as sns

from pylab import rcParams

from sklearn.model_selection import train_test_split

from keras.models import Model, load_model

from keras.layers import Input, Dense

from keras.callbacks import ModelCheckpoint, TensorBoard

from keras import regularizers



%matplotlib inline



sns.set(style='whitegrid', palette='muted', font_scale=1.5)



rcParams['figure.figsize'] = 14, 8



RANDOM_SEED = 42

LABELS = ["non_pulsar", "pulsar"]
print(df.shape)

print(df.isnull().values.any())
count_classes = pd.value_counts(df['target_class'], sort = True)

count_classes.plot(kind = 'bar', rot=0)

plt.title("Transaction class distribution")

plt.xticks(range(2), LABELS)

plt.xlabel("Class")

plt.ylabel("Frequency");
pulsar = df[df.target_class == 1]

non_pulsar = df[df.target_class == 0]



print(pulsar.shape)

print(non_pulsar.shape)

print('====================')

print('non pulsar')

print('====================')

print(non_pulsar.describe())

print('====================')

print('pulsar')

print('====================')

print(pulsar.describe())
from sklearn.preprocessing import StandardScaler



X_train, X_test = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)

X_train = X_train[X_train.target_class == 0]

X_train = X_train.drop(['target_class'], axis=1)



y_test = X_test['target_class']

X_test = X_test.drop(['target_class'], axis=1)



X_train = X_train.values

X_test = X_test.values

print(X_train[0])

print(X_train.shape)

print(X_test.shape)

scaler = StandardScaler()

scaler.fit(X_train)

scaler.fit(X_train)

StandardScaler(copy=True, with_mean=True, with_std=True)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)

print(X_train[0])
input_dim = X_train.shape[1]

encoding_dim = 14



input_layer = Input(shape=(input_dim, ))



encoder = Dense(encoding_dim, activation="tanh", 

                activity_regularizer=regularizers.l1(10e-5))(input_layer)

encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)



decoder = Dense(int(encoding_dim / 2), activation='tanh')(encoder)

decoder = Dense(input_dim, activation='relu')(decoder)



autoencoder = Model(inputs=input_layer, outputs=decoder)
nb_epoch = 1000

batch_size = 320



autoencoder.compile(optimizer='sgd', 

                    loss='mean_squared_error', 

                    metrics=['accuracy'])



checkpointer = ModelCheckpoint(filepath="model.h5",

                               verbose=0,

                               save_best_only=True)

tensorboard = TensorBoard(log_dir='./logs',

                          histogram_freq=0,

                          write_graph=True,

                          write_images=True)



history = autoencoder.fit(X_train, X_train,

                    epochs=nb_epoch,

                    batch_size=batch_size,

                    shuffle=True,

                    validation_data=(X_test, X_test),

                    verbose=1,

                    callbacks=[checkpointer, tensorboard]).history
plt.plot(history['loss'])

plt.plot(history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper right');

plt.show()
predictions = autoencoder.predict(X_test)



mse = np.mean(np.power(X_test - predictions, 2), axis=1)

error_df = pd.DataFrame({'reconstruction_error': mse,

                        'true_class': y_test})



error_df.describe()
fig = plt.figure()

ax = fig.add_subplot(111)

fraud_error_df = error_df[error_df['true_class'] == 1]

_ = ax.hist(fraud_error_df.reconstruction_error.values, bins=10)
from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,

                             roc_curve, recall_score, classification_report, f1_score,

                             precision_recall_fscore_support)
fpr, tpr, thresholds = roc_curve(error_df.true_class, error_df.reconstruction_error)

roc_auc = auc(fpr, tpr)



plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, label='AUC = %0.4f'% roc_auc)

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.001, 1])

plt.ylim([0, 1.001])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
precision, recall, th = precision_recall_curve(error_df.true_class, error_df.reconstruction_error)

plt.plot(recall, precision, 'b', label='Precision-Recall curve')

plt.title('Recall vs Precision')

plt.xlabel('Recall')

plt.ylabel('Precision')

plt.show()
plt.plot(th, precision[1:], 'b', label='Threshold-Precision curve')

plt.title('Precision for different threshold values')

plt.xlabel('Threshold')

plt.ylabel('Precision')

plt.show()
plt.plot(th, recall[1:], 'b', label='Threshold-Recall curve')

plt.title('Recall for different threshold values')

plt.xlabel('Reconstruction error')

plt.ylabel('Recall')

plt.show()
threshold = 1.3

groups = error_df.groupby('true_class')

fig, ax = plt.subplots()



for name, group in groups:

    ax.plot(group.index, group.reconstruction_error, marker='o', ms=3.5, linestyle='',

            label= "pulsar" if name == 1 else "non_pulsar")

ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')

ax.legend()

plt.title("Reconstruction error for different classes")

plt.ylabel("Reconstruction error")

plt.xlabel("Data point index")

plt.show();
y_pred = [1 if e > threshold else 0 for e in error_df.reconstruction_error.values]

conf_matrix = confusion_matrix(error_df.true_class, y_pred)



plt.figure(figsize=(12, 12))

sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");

plt.title("Confusion matrix")

plt.ylabel('True class')

plt.xlabel('Predicted class')

plt.show()