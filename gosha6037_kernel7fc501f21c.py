import numpy as np
import pandas as pd
import keras
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt
import os
import re
print(os.listdir("../input"))
def to_int(obj):
    return int(re.sub("[^\d]", '', obj))
df = pd.read_csv('../input/2TWH_train.csv', index_col='IDNum')
for col in df:
    if col[0] == ' ':
        df = df.rename(columns={col : col[1:]})
dt = pd.read_csv('../input/test.csv', index_col='IDNum')
for col in dt:
    if col[0] == ' ':
        dt = dt.rename(columns={col : col[1:]})
df['Source IP'] = df['Source IP'].apply(to_int)
df['Destination IP'] = df['Destination IP'].apply(to_int)
df['Timestamp'] = df['Timestamp'].apply(to_int)
df['Flow Bytes/s'] = df['Flow Bytes/s'].astype(float) 
df['Flow Packets/s'] = df['Flow Packets/s'].astype(float) 

dt['Source IP'] = dt['Source IP'].apply(to_int)
dt['Destination IP'] = dt['Destination IP'].apply(to_int)
dt['Timestamp'] = dt['Timestamp'].apply(to_int)
dt['Flow Bytes/s'] = dt['Flow Bytes/s'].astype(float) 
dt['Flow Packets/s'] = dt['Flow Packets/s'].astype(float)
MaxValueCount = 100000
df = df.drop(['Private'], axis='columns')
UselessAttribute = set(['Flow ID'])
for i in range(df.shape[1]):
    value_count = len(df[df.columns[i]].value_counts())
    if (value_count == 1 or value_count > MaxValueCount):
        UselessAttribute.add(df.columns[i])
for attribute in UselessAttribute:
    df = df.drop(attribute, axis='columns')
    dt = dt.drop(attribute, axis='columns')
UselessAttribute.clear()
for i in range(dt.shape[1]):
    value_count = len(dt[dt.columns[i]].value_counts())
    if (value_count == 1 or value_count > MaxValueCount):
        UselessAttribute.add(dt.columns[i])
for attribute in UselessAttribute:
    df = df.drop(attribute, axis='columns')
    dt = dt.drop(attribute, axis='columns')
del UselessAttribute
df = df.replace([np.inf, 'Infinity', 'infinity', 'inf'], 2**31-1)
df = df.replace([np.nan, np.inf, 'NaN'], 0)
dt = dt.replace([np.inf, 'Infinity', 'infinity', 'inf'], 2**31-1)
dt = dt.replace([np.nan, 'NaN'], 0)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
enc = LabelEncoder()
y = enc.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)
ohe = OneHotEncoder(categories='auto')
y_train_ohe = ohe.fit_transform(y_train.reshape(-1, 1))
y_test_ohe = ohe.fit_transform(y_test.reshape(-1, 1))
y_ohe = ohe.fit_transform(y.reshape(-1, 1))
scaler = StandardScaler()
X_scale = scaler.fit_transform(X.astype(float))
X_train_scale = scaler.fit_transform(X_train.astype(float))
X_test_scale = scaler.transform(X_test.astype(float))
from keras import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(units=20, activation='relu', input_dim=X.shape[1]))
model.add(Dense(units=10, activation='relu'))
model.add(Dense(units=3, activation='softmax'))
model.compile(optimizer='rmsprop',
             loss='categorical_crossentropy',
             metrics=['accuracy'])
history = model.fit(X_train_scale,
                   y_train_ohe,
                   epochs=1000,
                   batch_size=8192,
                   validation_data=(X_test_scale, y_test_ohe))
import matplotlib.pyplot as plt
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.figure(figsize = (10, 10))
plt.semilogy(epochs, loss, 'bo', label='Training loss')
plt.semilogy(epochs, val_loss, 'red', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.clf()
history_dict = history.history
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']
plt.figure(figsize = (10, 10))
plt.semilogy(epochs, acc_values, 'bo', label='Training acc')
plt.semilogy(epochs, val_acc_values, 'red', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
from sklearn.metrics import confusion_matrix
confusion_matrix(y, model.predict_classes(X))
dt_scale = scaler.transform(dt)
dftest = pd.DataFrame(enc.inverse_transform(model.predict_classes(dt_scale)), columns=['Label'], index=dt.index).reset_index()
dftest.sort_values('IDNum')
dftest.to_csv('submission.csv', index=False)