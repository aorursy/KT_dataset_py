# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import keras

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MaxAbsScaler

from keras.models import Sequential

from keras.layers import Dense, Dropout

import matplotlib.pyplot as plt

from IPython.display import FileLink
train_df = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

X = train_df.iloc[:, 1:]

y = train_df.iloc[:, 0]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
# Aqui não foi necessario a camada flatten, pois os dados ja estão "flatten"

model = Sequential()

model.add( Dense(512, activation="relu", input_shape=(784, )) ) 

model.add( Dense(256, activation="relu") ) 

model.add( Dropout(0.2) ) 

model.add( Dense(10, activation="softmax") ) 
model.compile(

    optimizer="adam",

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)
pixeScaler = MaxAbsScaler()

X_train_scaled = pixeScaler.fit_transform(X_train)
epochs = 5

history = model.fit(

    X_train_scaled, y_train,

    epochs=epochs,

    validation_split=0.2

)
(perda_teste, acuracia_teste) = model.evaluate(pixeScaler.transform(X_test), y_test)



print("Perda do teste:", perda_teste)

print("Acurácia do teste:", acuracia_teste)
test_df = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

test_df.shape
label_id = test_df.index.values + 1

classes = model.predict_classes(pixeScaler.transform(test_df))
submission_df = pd.DataFrame({"ImageId": label_id, "Label": classes})

submission_df.head()
sample_df = pd.read_csv("/kaggle/input/digit-recognizer/sample_submission.csv")

sample_df.head()
#submission_df.to_csv("submision.csv", index=False)

#FileLink('submision.csv')