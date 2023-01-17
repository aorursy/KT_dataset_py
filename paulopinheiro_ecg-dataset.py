# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd
train_df = pd.read_csv('/kaggle/input/heartbeat/mitbih_train.csv', header=None)

test_df  = pd.read_csv('/kaggle/input/heartbeat/mitbih_test.csv', header=None)
train_df[187] = train_df[187].astype(int)



categories_counts = train_df[187].value_counts()

print(categories_counts)
import matplotlib.pyplot as plt



plt.figure(figsize=(5,5))

plt.pie(categories_counts, labels=['n','q','v','s','f'], colors=['red','green','blue','skyblue','orange'], autopct='%1.1f%%')

plt.show()
from sklearn.utils import resample



df_0 = train_df[train_df[187]==0]

df_1 = train_df[train_df[187]==1]

df_2 = train_df[train_df[187]==2]

df_3 = train_df[train_df[187]==3]

df_4 = train_df[train_df[187]==4]



df_0_downsample = resample(df_0,replace=True,n_samples=5000,random_state=122)

df_1_upsample   = resample(df_1,replace=True,n_samples=5000,random_state=123)

df_2_upsample   = resample(df_2,replace=True,n_samples=5000,random_state=124)

df_3_upsample   = resample(df_3,replace=True,n_samples=5000,random_state=125)

df_4_upsample   = resample(df_4,replace=True,n_samples=5000,random_state=126)



train_df=pd.concat([df_0_downsample,df_1_upsample,df_2_upsample,df_3_upsample,df_4_upsample])
plt.figure(figsize=(10,5))



plt.subplot(2,2,1)

plt.plot(df_0.iloc[100,:186])



plt.subplot(2,2,2)

plt.plot(df_1.iloc[100,:186])



plt.subplot(2,2,3)

plt.plot(df_2.iloc[100,:186])



plt.subplot(2,2,4)

plt.plot(df_3.iloc[100,:186])



plt.show()
from keras.utils.np_utils import to_categorical



target_train = train_df[187]

target_test  = test_df[187]

y_train = to_categorical(target_train)

y_test  = to_categorical(target_test)
X_train = train_df.iloc[:,:186].values

X_test  = test_df.iloc[:,:186].values



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

model.add(layers.Dense(5, activation='softmax'))



print(model.summary())
from keras.callbacks import EarlyStopping

callbacks = [EarlyStopping(monitor='val_loss', patience=8)]
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
history = model.fit(X_train, y_train, callbacks=[], validation_data=(X_test, y_test), epochs = 30, batch_size = 64)
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

                     index   = ['N', 'S', 'V', 'F', 'Q'], 

                     columns = ['N', 'S', 'V', 'F', 'Q'])



plt.figure(figsize=(10,10))

sns.heatmap(cm_df, annot=True, fmt="d", linewidths=0.5, cmap='Blues', cbar=False, annot_kws={'size':14}, square=True)

plt.title('Kernel \nAccuracy:{0:.3f}'.format(accuracy_score(y_test_category, y_pred)))

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()
from sklearn.metrics import classification_report

print(classification_report(y_test_category, y_pred, target_names=['N', 'S', 'V', 'F', 'Q']))
from tqdm import notebook



def prep_data(data_frame):

    df_list = []

    label_list=[]

    for k in notebook.tqdm(range(len(data_frame))):

    # for k in range(2):

        df = pd.DataFrame(data_frame.iloc[k,:186])

        df.columns = ["values"]

        df['id'] = k

        df['time'] = df.index

        df_list.append(df)

        

        df_label = pd.DataFrame(columns=['id', 'values'], data=[[k, data_frame.iloc[k,187].astype(int)]])

        label_list.append(df_label)



    df = pd.concat(df_list, ignore_index = True, sort = False)

    df_label = pd.concat(label_list, ignore_index = True, sort = False)

    

    return df, df_label
X_train,y_train = prep_data(train_df)
X_test,y_test = prep_data(test_df)
from tsfresh import extract_features

from tsfresh import extract_relevant_features

from tsfresh.feature_extraction.settings import from_columns

from tsfresh.feature_extraction.settings import ComprehensiveFCParameters, MinimalFCParameters, EfficientFCParameters
ComprehensiveFCParameters()
extraction_settings = dict({'median': None, 'mean': None, 'standard_deviation': None, 'variance': None, 'abs_energy': None, 'skewness': None, 'kurtosis': None, 'sample_entropy': None,  

                            'spkt_welch_density': [{'coeff': 2}, {'coeff': 5}, {'coeff': 8}],

                            'time_reversal_asymmetry_statistic': [{'lag': 1}, {'lag': 2}, {'lag': 3}],

                            'fft_aggregated': [{'aggtype': 'centroid'},{'aggtype': 'variance'},{'aggtype': 'skew'},{'aggtype': 'kurtosis'}]})
X_train_features = extract_features(X_train, column_id='id', column_sort='time', default_fc_parameters=extraction_settings)
X_test_features = extract_features(X_test, column_id='id', column_sort='time', default_fc_parameters=extraction_settings)
from sklearn import preprocessing

min_max_scaler   = preprocessing.MinMaxScaler()

X_train_features = min_max_scaler.fit_transform(X_train_features)

X_test_features  = min_max_scaler.transform(X_test_features)
model = models.Sequential()

model.add(layers.GaussianNoise(0.01, input_shape=(X_train_features.shape[1],)))

model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dropout(0.2))

model.add(layers.BatchNormalization())

model.add(layers.Dense(32, activation='relu'))

model.add(layers.Dropout(0.2))

model.add(layers.BatchNormalization())

model.add(layers.Dense(16, activation='relu'))

model.add(layers.Dropout(0.2))

model.add(layers.BatchNormalization())

model.add(layers.Dense(5, activation='softmax'))



print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
history = model.fit(X_train_features, to_categorical(y_train['values']), callbacks=[], validation_data=(X_test_features, to_categorical(y_test['values'])), epochs = 40, batch_size = 128)
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

import seaborn as sns



y_pred = model.predict_classes(X_test_features)



y_test_category = y_test['values']  



# Creates a confusion matrix

cm = confusion_matrix(y_test_category, y_pred) 



# Transform to df for easier plotting

cm_df = pd.DataFrame(cm,

                     index   = ['N', 'S', 'V', 'F', 'Q'], 

                     columns = ['N', 'S', 'V', 'F', 'Q'])



plt.figure(figsize=(10,10))

sns.heatmap(cm_df, annot=True, fmt="d", linewidths=0.5, cmap='Blues', cbar=False, annot_kws={'size':14}, square=True)

plt.title('Kernel \nAccuracy:{0:.3f}'.format(accuracy_score(y_test_category, y_pred)))

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()