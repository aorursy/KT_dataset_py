import tensorflow as tf

from tensorflow import keras

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split



%matplotlib inline

plt.style.use("ggplot")
df_train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

df_test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
df_train.head()
print(df_train.shape)

print(df_test.shape)
plt.figure(figsize=(10,5))

sns.countplot(df_train["label"])

plt.show()
X = df_train.iloc[:,1:]

y = df_train.iloc[:,:1]
print(X.shape)

print(y.shape)
X = X / 255.0

df_test = df_test / 255.0
X.head(2)
## Reshape The Data

X = X.values.reshape(-1,28,28,1)

df_test = df_test.values.reshape(-1,28,28,1)
y.head()
y = keras.utils.to_categorical(y,num_classes=10)

y
## Split The Data into X_train, X_test, y_train, y_test

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1)
print(f"X_train.shape : {X_train.shape}")

print(f"X_test.shape : {X_test.shape}")

print(f"y_train.shape : {y_train.shape}")

print(f"y_test.shape : {y_test.shape}")
model = keras.Sequential([

    keras.layers.Conv2D(32,(3,3),input_shape=(28,28,1)),

    keras.layers.MaxPooling2D(2,2),

    

    keras.layers.Conv2D(64,(3,3),activation="relu"),

    keras.layers.MaxPooling2D(2,2),

    

    keras.layers.Conv2D(64,(3,3),activation="relu"),

    keras.layers.MaxPooling2D(2,2),

    

    keras.layers.Dropout(0.25),

    

    keras.layers.Flatten(),

    keras.layers.Dense(128,activation="relu"),

    keras.layers.Dropout(0.5),

    keras.layers.Dense(10,activation="softmax")

])
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
best_model = keras.callbacks.ModelCheckpoint("best_model.h5",save_best_only=True)

history = model.fit(X_train,y_train,epochs=50,verbose=1,validation_data=(X_test,y_test),callbacks=[best_model])
plt.figure(figsize=(10,5))



acc = history.history["accuracy"]

val_acc = history.history["val_accuracy"]

loss = history.history["loss"]

val_loss = history.history["val_loss"]



epochs = range(len(acc))



plt.plot(epochs,acc,"r",label="Training Accuracy")

plt.plot(epochs,val_acc,"b",label="Validation Accuracy")



plt.legend()

plt.figure()



plt.figure(figsize=(10,5))



plt.plot(epochs,loss,"r",label="Training Loss")

plt.plot(epochs,val_loss,"b",label="Validation Loss")



plt.legend()

plt.show()