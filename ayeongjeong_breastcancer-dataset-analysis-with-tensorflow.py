import numpy as np

import pandas as pd

from sklearn import datasets

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler
cancer = datasets.load_breast_cancer()

cancer.keys()
print(cancer.data.shape)

print(np.unique(cancer.target, return_counts=True))
sns.countplot(cancer.target)
df_cancer = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])

df_cancer['target'] = cancer['target']

df_cancer.head()
plt.figure(figsize=(15,5))

plt.boxplot(cancer.data)

plt.xlabel('feature')

plt.ylabel('value')

plt.show()
cancer.feature_names[[3,23]]
scaler = StandardScaler()

X = cancer.data

scaled_X = scaler.fit_transform(X) 
plt.figure(figsize=(15,5))

plt.boxplot(scaled_X)

plt.xlabel('feature')

plt.ylabel('value')

plt.show()
for i in range(len(scaled_X)):

    scaled_X[i][scaled_X[i]>6] = scaled_X[i].mean()
plt.figure(figsize=(15,5))

plt.boxplot(scaled_X)

plt.xlabel('feature')

plt.ylabel('value')

plt.show()
import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers

from tensorflow.keras import Sequential

from sklearn.model_selection import train_test_split
# 데이터셋 생성

x_train, x_test, y_train, y_test = train_test_split(scaled_X, cancer.target, test_size=0.3)



# 모델 구성

model = keras.Sequential([

                          layers.Input(shape=x_train.shape[1]),

                          layers.Dense(8, activation='relu'),

                          layers.Dense(1, activation='sigmoid')

])

# 모델 학습 과정 설정

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
hist = model.fit(x_train, y_train, epochs=50, validation_split=0.3)
model.evaluate(x_test, y_test)
plt.plot(hist.history['loss'],'x--',label='loss')

plt.plot(hist.history['val_loss'],'x--',label='val_loss')

plt.plot(hist.history['val_accuracy'],'-',label='val_accuracy')

plt.plot(hist.history['accuracy'],'-',label='accuracy')

plt.legend()

plt.show()