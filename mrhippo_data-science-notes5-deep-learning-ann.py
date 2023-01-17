import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, Input, ReLU, LeakyReLU, BatchNormalization
from keras.models import Model, Sequential
from keras.optimizers import Adam

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv("/kaggle/input/breast-cancer-wisconsin-data/data.csv")
df.head()
df["diagnosis"].unique()
df.info()
df = df.drop(["id","Unnamed: 32"],axis = 1) #these columns are irrelevant for model
df.describe()
f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(df.corr(),linecolor = "black", linewidths=0.5, fmt = '.2f', ax=ax,cmap = "BrBG")
plt.title("Correlations",fontsize = 18)
plt.show()
fig = plt.figure(figsize = (12,8))
sns.countplot(df["diagnosis"])
plt.title("Diagnosis Count")
plt.show()
# preparing data parts for boxplot visualization
df_for_box1 = df[df.columns[1:11]]
df_for_box2 = df[df.columns[11:21]]
df_for_box3 = df[df.columns[21:31]]
df_for_box1.head()
df_for_box2.head()
df_for_box3.head()
plt.figure(figsize = (15,8))
sns.boxplot(data=df_for_box1, orient="h", palette="Set2")
plt.title("First 10 Columns")
plt.show()
plt.figure(figsize = (15,8))
sns.boxplot(data=df_for_box2, orient="h", palette="Set2")
plt.title("Second First 10 Columns")
plt.show()
plt.figure(figsize = (15,8))
sns.boxplot(data=df_for_box3, orient="h", palette="Set2")
plt.title("Third First 10 Columns")
plt.show()
df["diagnosis"] = [1 if each == "M" else 0 for each in df["diagnosis"]] # M -> 1, B -> 0
data_target = df["diagnosis"]
data_values = df.drop(["diagnosis"],axis = 1)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data_values,data_target,test_size = 0.2, random_state = 42)
print("x_train shape: ",x_train.shape)
print("y_train shape: ",y_train.shape)
print("x_test shape: ",x_test.shape)
print("y_test shape: ",y_test.shape)
y_train = y_train.values.reshape(-1,1)
y_test = y_test.values.reshape(-1,1)

print("changed y_train shape for keras",y_train.shape)
print("changed y_test shape for keras",y_test.shape)
from sklearn.preprocessing import normalize
x_train = normalize(x_train)
x_test = normalize(x_test)
from sklearn.preprocessing import StandardScaler
x_train = StandardScaler().fit_transform(x_train)
x_test = StandardScaler().fit_transform(x_test)
model = Sequential()

model.add(Dense(128, input_shape = (455, 30)))
model.add(ReLU())
model.add(Dropout(0.1))

model.add(Dense(64))
model.add(ReLU())
model.add(Dropout(0.1))

model.add(Dense(1, activation = "sigmoid"))

model.compile(loss = "binary_crossentropy",
             optimizer = "adam",
             metrics = ["accuracy"])

history = model.fit(x_train,y_train,epochs = 20, batch_size = 5,validation_data = (x_test,y_test))
history.history.keys()
plt.figure(figsize = (13,8))
plt.plot(history.history["accuracy"], label = "Train Accuracy")
plt.plot(history.history["val_accuracy"],label = "Validation Accuracy")
plt.title("Accuracies")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
plt.figure(figsize = (13,8))
plt.plot(history.history["loss"], label = "Train Loss")
plt.plot(history.history["val_loss"],label = "Validation Loss")
plt.title("Losses")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
df = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
df.head()
fig = plt.figure(figsize = (12,8))
sns.countplot(df["label"],palette = "Blues")
plt.title("Label Count")
plt.show()
df_target = df["label"]
df_values = df.drop(["label"], axis = 1)/255.0 # normalization
x_train, x_test, y_train, y_test = train_test_split(df_values,df_target,test_size = 0.2, random_state = 42)
from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train,num_classes = 10)
y_test = to_categorical(y_test,num_classes = 10)
print("x_train shape: ",x_train.shape)
print("y_train shape: ",y_train.shape)
print("x_test shape: ",x_test.shape)
print("y_test shape: ",y_test.shape)
model = Sequential()

model.add(Dense(256, input_shape = (33600, 784)))
model.add(ReLU())
model.add(Dropout(0.2))

model.add(Dense(128))
model.add(ReLU())
model.add(Dropout(0.1))

model.add(Dense(64))
model.add(ReLU())
model.add(Dropout(0.1))

model.add(Dense(10, activation = "softmax"))

model.compile(loss = "categorical_crossentropy",
             optimizer = "adam",
             metrics = ["accuracy"])

history = model.fit(x_train,y_train,epochs = 30, batch_size = 100,validation_data = (x_test,y_test))
history.history.keys()
plt.figure(figsize = (13,8))
plt.plot(history.history["accuracy"], label = "Train Accuracy")
plt.plot(history.history["val_accuracy"],label = "Validation Accuracy")
plt.title("Accuracies")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
plt.figure(figsize = (13,8))
plt.plot(history.history["loss"], label = "Train Loss")
plt.plot(history.history["val_loss"],label = "Validation Loss")
plt.title("Losses")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()