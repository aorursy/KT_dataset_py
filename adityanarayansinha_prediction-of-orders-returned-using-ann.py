import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

dataset = pd.read_csv('../input/Training_Dataset_v2.csv')
dataset.head()
dataset.tail()
from sklearn.impute import SimpleImputer

dataset["lead_time"] = SimpleImputer(strategy = "median").fit_transform(dataset["lead_time"].values.reshape(-1,1))
dataset.head()

dataset = dataset.dropna()
for col in ['perf_6_month_avg','perf_12_month_avg']:

    dataset[col] = SimpleImputer(missing_values=-99,strategy='mean').fit_transform(dataset[col].values.reshape(-1,1))
dataset.head()
for col in ['potential_issue', 'deck_risk', 'oe_constraint', 'ppap_risk',

               'stop_auto_buy', 'rev_stop', 'went_on_backorder']:

        dataset[col] = (dataset[col] == 'Yes').astype(int)
dataset.head()
from sklearn.preprocessing import normalize

qty_related = ['national_inv', 'in_transit_qty', 'forecast_3_month', 

                   'forecast_6_month', 'forecast_9_month', 'min_bank',

                   'local_bo_qty', 'pieces_past_due', 'sales_1_month', 

                   'sales_3_month', 'sales_6_month', 'sales_9_month',]

dataset[qty_related] = normalize(dataset[qty_related], axis=1)
dataset.head()
X = dataset.iloc[:, :-1].values

y = dataset.iloc[:, -1].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense


ann = tf.keras.models.Sequential()


ann.add(tf.keras.layers.Dense(units=12, activation='relu'))


ann.add(tf.keras.layers.Dense(units=15, activation='relu'))


ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


ann.fit(X_train,y_train,batch_size=32,epochs=5)
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)
from sklearn.metrics import accuracy_score

print(accuracy_score(y_pred,y_test))