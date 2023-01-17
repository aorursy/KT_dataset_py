# Generic

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os, warnings, gc

warnings.filterwarnings("ignore")



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import train_test_split
url = '../input/all-datasets-for-practicing-ml/Class/Class_Abalone.csv'

data = pd.read_csv(url, header='infer')

print("Total Records: ", data.shape[0])
# Inspect

data.head()
fig_size = plt.rcParams["figure.figsize"]

fig_size[0] = 15

fig_size[1] = 10

plt.rcParams["figure.figsize"] = fig_size



data.Sex.value_counts().plot(kind='pie', autopct='%1.0f%%', colors=['beige','khaki','wheat'])
# Scaling

sc = StandardScaler()

cols = data.columns [1:]



for col in cols:

    data[col] = sc.fit_transform(data[col].values.reshape(-1,1))

    



# Encoding

encoder = LabelEncoder()

data['Sex'] = encoder.fit_transform(data['Sex'])



# Inspect

data.head()
# Libraries

import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, Activation, Dropout

from tensorflow.keras.models import Model
# Feature & Target Selection

columns = data.columns

target = ['Sex']

features = columns[1:]



X = data[features]

y = data[target]



# Dataset Split 

''' Training = 90% & Validation = 10%  '''

test_size = 0.1

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=True) 

# Build Model

input_layer = Input(shape=(X.shape[1],))

dense_layer1 = Dense(256, activation='relu')(input_layer)

dense_layer2 = Dense(128, activation='relu')(dense_layer1)

dense_layer3 = Dense(64, activation='relu')(dense_layer2)

output = Dense(y.shape[1], activation='softmax')(dense_layer3)



model = Model(inputs=input_layer, outputs=output)

model.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=['acc'])



model.summary()
# Train Model

history = model.fit(X_train, y_train, batch_size=8, epochs=50, verbose=1, validation_split=0.1)
#Evaluation

score = model.evaluate(X_val, y_val, verbose=0)



print("TensorFlow Model Accuracy:", '{:.2%}'.format(score[1]))