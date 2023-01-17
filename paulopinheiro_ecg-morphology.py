import warnings

warnings.filterwarnings('ignore')

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.linear_model import SGDClassifier

from sklearn.multiclass import OneVsOneClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import LinearSVC, SVC

from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import VotingClassifier



import seaborn as sn



hbeat_signals = pd.read_csv("../input/mitbih-arrhythmia-database-de-chazal-class-labels/DS1_signals.csv", header=None)

hbeat_labels = pd.read_csv("../input/mitbih-arrhythmia-database-de-chazal-class-labels//DS1_labels.csv", header=None)



print("+"*50)

print("Signals Info:")

print("+"*50)

print(hbeat_signals.info())

print("+"*50)

print("Labels Info:")

print("+"*50)

print(hbeat_labels.info())

print("+"*50)
hbeat_signals.head()
hbeat_signals.describe()
hbeat_signals = hbeat_signals.sub(0.5, axis=0)

hbeat_signals.describe()
# Collect data of different hheartbeats in different lists

#class 0

cl_0_idx = hbeat_labels[hbeat_labels[0] == 0].index.values

cl_N = hbeat_signals.iloc[cl_0_idx]

#class 1

cl_1_idx = hbeat_labels[hbeat_labels[0] == 1].index.values

cl_S = hbeat_signals.iloc[cl_1_idx]

#class 2

cl_2_idx = hbeat_labels[hbeat_labels[0] == 2].index.values

cl_V = hbeat_signals.iloc[cl_2_idx]

#class 3

cl_3_idx = hbeat_labels[hbeat_labels[0] == 3].index.values

cl_F = hbeat_signals.iloc[cl_3_idx]



# make plots for the different hbeat classes

plt.subplot(221)

for n in range(3):

    cl_N.iloc[n].plot(title='Class N (0)', figsize=(10,8))

plt.subplot(222)

for n in range(3):

    cl_S.iloc[n].plot(title='Class S (1)')

plt.subplot(223)

for n in range(3):

    cl_V.iloc[n].plot(title='Class V (2)')

plt.subplot(224)

for n in range(3):

    cl_F.iloc[n].plot(title='Class F (3)')

#check if missing data

print("Column\tNr of NaN's")

print('+'*50)

for col in hbeat_signals.columns:

    if hbeat_signals[col].isnull().sum() > 0:

        print(col, hbeat_signals[col].isnull().sum()) 

joined_data = hbeat_signals.join(hbeat_labels, rsuffix="_signals", lsuffix="_labels")



#rename columns

joined_data.columns = [i for i in range(180)]+['class']
joined_data.head()
joined_data.describe()
categories_counts = joined_data['class'].value_counts()

print(categories_counts)
print("class\t%")

joined_data['class'].value_counts()/len(joined_data)
from sklearn.model_selection import StratifiedShuffleSplit



split = StratifiedShuffleSplit(n_splits=1, test_size=0.2,random_state=42)



for train_index, test_index in split.split(joined_data, joined_data['class']):

    strat_train_set = joined_data.loc[train_index]

    strat_test_set = joined_data.loc[test_index]    
print("class\t%")

strat_train_set['class'].value_counts()/len(strat_train_set)
print("class\t%")

strat_test_set['class'].value_counts()/len(strat_test_set)
train_df = strat_train_set

test_df  = strat_test_set
from sklearn.utils import resample



df_0 = train_df[train_df['class']==0]

df_1 = train_df[train_df['class']==1]

df_2 = train_df[train_df['class']==2]

df_3 = train_df[train_df['class']==3]



df_0_downsample = resample(df_0,replace=True,n_samples=10000,random_state=122)

df_1_upsample   = resample(df_1,replace=True,n_samples=10000,random_state=123)

df_2_upsample   = resample(df_2,replace=True,n_samples=10000,random_state=124)

df_3_upsample   = resample(df_3,replace=True,n_samples=10000,random_state=125)



train_df=pd.concat([df_0_downsample,df_1_upsample,df_2_upsample,df_3_upsample])
plt.figure(figsize=(10,5))



plt.subplot(2,2,1)

plt.plot(df_0.iloc[0,:180])



plt.subplot(2,2,2)

plt.plot(df_1.iloc[0,:180])



plt.subplot(2,2,3)

plt.plot(df_2.iloc[0,:180])



plt.subplot(2,2,4)

plt.plot(df_3.iloc[0,:180])



plt.show()
from keras.utils.np_utils import to_categorical



target_train = train_df['class']

target_test  = test_df['class']

y_train = to_categorical(target_train)

y_test  = to_categorical(target_test)
X_train = train_df.iloc[:,:180].values

X_test  = test_df.iloc[:,:180].values



X_train = X_train.reshape(len(X_train), X_train.shape[1], 1)

X_test  = X_test.reshape(len(X_test), X_test.shape[1], 1)
import keras

from keras import models

from keras import layers



model = models.Sequential()

model.add(layers.GaussianNoise(0.01, input_shape=(X_train.shape[1], X_train.shape[2])))



model.add(layers.Conv1D(64, 16, activation='relu'))

model.add(layers.BatchNormalization())

model.add(layers.MaxPool1D(pool_size=4, strides=2, padding="same"))



model.add(layers.Conv1D(64, 12, activation='relu'))

model.add(layers.BatchNormalization())

model.add(layers.MaxPool1D(pool_size=3, strides=2, padding="same"))



model.add(layers.Conv1D(64, 8, activation='relu'))

model.add(layers.BatchNormalization())

model.add(layers.MaxPool1D(pool_size=2, strides=2, padding="same"))



model.add(layers.Flatten())

model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dropout(0.1))

model.add(layers.Dense(32, activation='relu'))

model.add(layers.Dropout(0.1))

model.add(layers.Dense(4, activation='softmax'))



print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
from keras.callbacks import EarlyStopping

callbacks = [EarlyStopping(monitor='val_loss', patience=8)]



history = model.fit(X_train, y_train, callbacks=callbacks, validation_data=(X_test, y_test), epochs = 20, batch_size = 128)
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']



epochs = range(1, len(acc)+1)



plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.legend()
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

import seaborn as sns



y_pred = model.predict_classes(X_test)



y_test_category = y_test.argmax(axis=-1)



# Creates a confusion matrix

cm = confusion_matrix(y_test_category, y_pred) 



# Transform to df for easier plotting

cm_df = pd.DataFrame(cm,

                     index   = ['N', 'S', 'V', 'F', ], 

                     columns = ['N', 'S', 'V', 'F', ])



plt.figure(figsize=(10,10))

sns.heatmap(cm_df, annot=True, fmt="d", linewidths=0.5, cmap='Blues', cbar=False, annot_kws={'size':14}, square=True)

plt.title('Kernel \nAccuracy:{0:.3f}'.format(accuracy_score(y_test_category, y_pred)))

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()
from sklearn.metrics import classification_report

print(classification_report(y_test_category, y_pred, target_names=['N', 'S', 'V', 'F']))