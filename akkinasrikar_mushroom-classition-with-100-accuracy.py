import pandas as pd

import numpy as np

import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib import style

style.use("fivethirtyeight")

from time import time
mushroomdata=pd.read_csv("/kaggle/input/mushroom-classification/mushrooms.csv")

mushroomdata.head()
from sklearn.preprocessing import LabelEncoder
cols=mushroomdata.columns
encoding=LabelEncoder()
for i in cols:

    mushroomdata[i]=encoding.fit_transform(mushroomdata[i])
mushroomdata.head()
mushroomdata.corr()
mushroomdata.describe().transpose()

import tensorflow as tf

from tensorflow.keras.layers import Dense

from tensorflow.keras.models import Sequential

from tensorflow.keras.callbacks import EarlyStopping,TensorBoard

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,classification_report


x=mushroomdata.drop("class",axis=1).values

y=mushroomdata["class"].values

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=12)


def modelcreating(optimizer="adam",loss="binary_crossentropy"):

    model=Sequential()

    model.add(Dense(22,input_dim=22,activation="relu"))

    model.add(Dense(22,activation="relu"))

    model.add(Dense(11,activation="relu"))

    model.add(Dense(5,activation="relu"))

    model.add(Dense(1,activation="sigmoid"))

    model.compile(loss=loss,optimizer=optimizer,metrics=["accuracy"])

    return model

model=modelcreating()
earlystopping=EarlyStopping(monitor="val_loss",mode="min",patience=25)


tensorboard=TensorBoard(log_dir="logs\{}".format(time()),

                        histogram_freq=1,

                       write_graph=True,)



model.fit(x=x_train,y=y_train,

          validation_data=(x_test,y_test),

         epochs=100,batch_size=30,verbose=1,

         callbacks=[tensorboard])
loss=pd.DataFrame(model.history.history)

loss.head()
plt.figure(figsize=(10,6))

loss[["loss","val_loss"]].plot()
plt.figure(figsize=(10,6))

loss[["accuracy","val_accuracy"]].plot()
y_pred=model.predict_classes(x_test)



# analysing the results

print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))


scores=model.evaluate(x_test,y_test)

print(model.metrics_names[1],scores[1]*100)
# Load the TensorBoard notebook extension

%load_ext tensorboard

%tensorboard --logdir logs