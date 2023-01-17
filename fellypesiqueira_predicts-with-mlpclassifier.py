# dataset import 

import pandas as pd
import numpy as np

df_train = pd.read_csv('../input/desafio-worcap-2020/treino.csv') # Data frame of training
df_test = pd.read_csv('../input/desafio-worcap-2020/teste.csv') # Data frame of test

(df_train.shape, df_test.shape)
# Split of predictors and target

X_train = df_train.iloc[:,1:28] # Set of training entries
y_train = df_train['label'] # training target
X_test = df_test.drop('id', 1) # Set of test entries
# Pre-processing: Data normalization 

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1,1)) # Nomalization between (-1,1)
X_train_norm = scaler.fit_transform(X_train) 
X_test_norm = scaler.transform(X_test) 
# Model create and training

from sklearn.neural_network import MLPClassifier

model = MLPClassifier(hidden_layer_sizes=(29),
                      activation='tanh',
                      solver='sgd',
                      learning_rate='adaptive',
                      learning_rate_init=0.125,
                      momentum=0.93,
                      batch_size=32,
                      alpha=0.001,
                      random_state=2020,
                      verbose=True)

model.fit(X_train_norm, y_train)

accuracy = model.score(X_train_norm, y_train)
print(f'\n\nAccuracy --> {round(accuracy*100,2)}%')
# validation metrics

from sklearn.metrics import confusion_matrix

y_pred = model.predict(X_train_norm)

print(confusion_matrix(y_train, y_pred))
# Submit file

label_test = model.predict(X_test_norm)

submit = pd.DataFrame({'Id':df_test['id'],
                       'Label':label_test})

submit