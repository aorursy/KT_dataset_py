import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# tf section

import tensorflow as tf

from tensorflow import keras



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv("/kaggle/input/titanic/train.csv")

df
df = df[["Survived", "Pclass","Sex", "Age", "SibSp", "Parch", "Fare"]]

df
# male 1, female 0

df = df.copy()

for i in range(len(df)):

    if df.loc[i, "Sex"] == "male":

        df.loc[i, "Sex"] = 1

    else:

        df.loc[i, "Sex"] = 0

df.head()
df = df.dropna()
df.hist("Sex", "Survived", figsize=(5, 3))

plt.suptitle("Sex/Survived")

df.hist("Age", "Survived", figsize=(5, 3))

plt.suptitle("Age/Survived")

df.hist("Parch", "Survived", figsize=(5, 3))

plt.suptitle("Parch/Survived")
x = df[["Pclass","Sex", "Age", "SibSp", "Parch", "Fare"]]

y = df["Survived"]



x = np.asarray(x).astype(np.float32)

y = np.asarray(y).astype(np.float32)



x = keras.utils.normalize(x)



x[0].shape
model = keras.Sequential()



model.add(keras.layers.Dense(20, input_shape=x[0].shape))

model.add(keras.layers.Activation("relu"))

model.add(keras.layers.Dense(20))

model.add(keras.layers.Activation("relu"))

model.add(keras.layers.Dense(20))

model.add(keras.layers.Activation("relu"))

model.add(keras.layers.Dense(1))

model.add(keras.layers.Activation("softmax"))

model.compile(loss=keras.losses.MeanSquaredError(),

              optimizer=keras.optimizers.Adam(learning_rate=0.1))



history = model.fit(x, y, epochs=200)
z = model.predict(x)

cnt = 0



for i, j in zip(y, z):

    if i == j:

        cnt += 1

        

print(cnt/len(z))