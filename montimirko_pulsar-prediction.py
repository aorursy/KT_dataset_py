import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import tensorflow as tf

import tensorflow.keras as keras

from sklearn.model_selection import train_test_split



import matplotlib.pyplot as plt

import seaborn as sns

path = '/kaggle/input/predicting-a-pulsar-star/pulsar_stars.csv'

data = pd.read_csv(path)
print(data.isnull().sum())
lunghezza = len(data.iloc[1,:])-1



# Set figsize here

fig, axes = plt.subplots(nrows=8, ncols=1, figsize=(12,48))



# flatten axes for easy iterating

for i, ax in enumerate(axes.flatten()):

    xx = data.iloc[:,i]

    yy = data.iloc[:,1]

    sns.scatterplot(x=xx, y=yy, hue="target_class", data=data, ax=ax)



fig.tight_layout()

plt.matshow(data.corr())

plt.title('Correlation Matrix')

plt.show()
label = data['target_class']

train = data.drop( 'target_class', axis=1, inplace=False)



print(label.value_counts()) # 1 a 1 gli altri 10



train_data,  test_data, train_label, test_label =  train_test_split(data, label, test_size=0.1)



model = keras.models.Sequential()



model.add(keras.layers.Dense(9))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Activation('relu'))



model.add(keras.layers.Dense(64))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Activation('relu'))



model.add(keras.layers.Dense(256))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Activation('relu'))



model.add(keras.layers.Dense(64))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Activation('relu'))



model.add(keras.layers.Dense(1, activation='sigmoid'))





model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(np.array(train_data),np.array(train_label), epochs=15)
acc = history.history['acc']

epochs_=range(0,15)  



plt.plot(epochs_, acc, label='acc')

plt.xlabel('no of epochs')

plt.ylabel('acc')





plt.title("no of epochs vs accuracy")

plt.legend()
model.evaluate(test_data, test_label)
prova = model.predict(train_data)



pulsar_numer = 0

for i in prova:

    if i > 0.8:

        pulsar_numer = pulsar_numer +1

        



print(train_label.value_counts())

print(pulsar_numer)
import seaborn as sns



ax1 = sns.distplot(train_label, hist=False, color="r", label="Actual Value")

sns.distplot(model.predict(train_data), hist=False, color="b", label="Fitted Values" , ax=ax1)