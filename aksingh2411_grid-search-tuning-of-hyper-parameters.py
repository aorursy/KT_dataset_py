# Common imports

import pandas as pd

import numpy as np

import time

import os

import tensorflow as tf

import tensorflow_hub as hub

from tensorflow import keras
#Verifying pathname of dataset before loading - for Kaggle

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename));

        print(os.listdir("../input"))
# Load Datasets

def loadDataset(file_name):

    df = pd.read_csv(file_name,engine = 'python')

    return df

start_time= time.time()

df_train = loadDataset("/kaggle/input/dataset-of-malicious-and-benign-webpages/Webpages_Classification_train_data.csv/Webpages_Classification_train_data.csv")

df_test = loadDataset("/kaggle/input/dataset-of-malicious-and-benign-webpages/Webpages_Classification_test_data.csv/Webpages_Classification_test_data.csv")

#Ensuring correct sequence of columns 

df_train = df_train[['url','content','label']]

df_test = df_test[['url','content','label']]

print("***Total Time taken --- %s seconds ---***" % (time.time() - start_time))
start_time= time.time()

df_test['content'] = df_test['content'].str.lower()

df_test.drop(columns=['url'],inplace=True)

df_test.rename(columns={'content':'text'},inplace=True)

df_train['content'] = df_train['content'].str.lower()

df_train.drop(columns=['url'],inplace=True)

df_train.rename(columns={'content':'text'},inplace=True)

print("***Total Time taken --- %s seconds ---***" % (time.time() - start_time))
#Converting Label value to 0,1

start_time= time.time()

df_test['label'].replace(to_replace ="good", value =1, inplace=True)

df_train['label'].replace(to_replace ="good", value =1, inplace=True)

df_test['label'].replace(to_replace ="bad", value =0, inplace=True)

df_train['label'].replace(to_replace ="bad", value =0, inplace=True)

print("***Total Time taken --- %s seconds ---***" % (time.time() - start_time))
#Selection lower numbers as of now for fast testing

train= df_train.iloc[:300000,]

val= df_train.iloc[300001:310000,]

test= df_test.iloc[:200000,]

print(len(train), 'train examples')

print(len(val), 'validation examples')

print(len(test), 'test examples')
#Converting the dataframes into X, y numpy arrays 

X_train = train['text'].to_numpy()

y_train = train['label'].astype(int).to_numpy()

X_val = val['text'].to_numpy()

y_val = val['label'].astype(int).to_numpy()

X_test = test['text'].to_numpy()

y_test = test['label'].astype(int).to_numpy()
# Using Transfer Learning from Tensorflow hub- Universal Text Encoder

start_time= time.time()

# Word Embedder with fixed 20 vector output

encoder = hub.load("https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1")

# Use the ecoder from a local file

#encoder = hub.load("datasets/PretrainedTFModel/1")

print("***Total Time taken --- %s seconds ---***" % (time.time() - start_time))
# Use scikit-learn to grid search 

import numpy

from sklearn.model_selection import GridSearchCV

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# Function to create model, required for KerasClassifier

def create_model(optimizer='adam'):

    model = keras.Sequential([

    hub.KerasLayer(encoder, input_shape=[],dtype=tf.string,trainable=True),

    keras.layers.Dense(32, activation='relu'),

    keras.layers.Dense(16, activation='relu'),

    keras.layers.Dense(1, activation='sigmoid'),

    ])

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

# create model

model = KerasClassifier(build_fn=create_model, epochs=4, batch_size=2048)

# define the grid search parameters

optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adam']

#optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']

param_grid = dict(optimizer=optimizer)

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1,cv=3)

grid_result = grid.fit(X_train,y_train)

# summarize results

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))
# Use scikit-learn to grid search the Learning Rate of ADAM Optimizer

import numpy

from sklearn.model_selection import GridSearchCV

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# Function to create model, required for KerasClassifier

def create_model(learning_rate=0.001):

    model = keras.Sequential([

    hub.KerasLayer(encoder, input_shape=[],dtype=tf.string,trainable=True),

    keras.layers.Dense(32, activation='relu'),

    keras.layers.Dense(16, activation='relu'),

    keras.layers.Dense(1, activation='sigmoid'),

    ])

    model.compile(loss='binary_crossentropy',optimizer=keras.optimizers.Adam(learning_rate=learning_rate), metrics=['accuracy'])

    return model

# create model

model = KerasClassifier(build_fn=create_model, epochs=5, batch_size=2048,verbose=0)

# define the grid search parameters

learning_rate = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001,0.000001,0.0000001]

param_grid = dict(learning_rate=learning_rate)

grid = GridSearchCV(estimator=model,param_grid=param_grid,n_jobs=1,cv=3)

grid_result = grid.fit(X_train,y_train)
# summarize results

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))