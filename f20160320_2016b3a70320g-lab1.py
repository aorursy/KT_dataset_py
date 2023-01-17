# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.utils import class_weight



from keras.layers import Dense, Dropout, BatchNormalization

from keras.models import Sequential

from keras.callbacks import EarlyStopping, ModelCheckpoint



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



# Seed value

# Apparently you may use different seed values at each stage

seed_value= 42



# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value

import os

os.environ['PYTHONHASHSEED']=str(0)



# 2. Set the `python` built-in pseudo-random generator at a fixed value

import random

random.seed(seed_value)



# 3. Set the `numpy` pseudo-random generator at a fixed value

import numpy as np

np.random.seed(seed_value)



# 4. Set the `tensorflow` pseudo-random generator at a fixed value

import tensorflow as tf

tf.random.set_seed(seed_value)

# for later versions: tf.random.set_random_seed(seed_value)

df = pd.read_csv("/kaggle/input/bitsf312-lab1/train.csv", index_col=["ID"], na_values='?')

df.head()
df['Class'].unique()
df.isna().sum()
df.dropna(inplace=True)

df.info()
df['Class'].value_counts()
df = pd.get_dummies(df)

y_lab = df['Class']

y = pd.get_dummies(df['Class'])

X = df.drop(['Class'], axis=1)
'''corr = df.corr()



fig, ax = plt.subplots(figsize=(20,10))

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

square=True, ax=ax, annot = True)'''
dropCols = ['Number of Insignificant Quantities','Number of Sentences', 'Total Number of Words']

X = X.drop(dropCols, axis = 1)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test, train_val, test_val =train_test_split(X, y, y_lab)

x_train.shape, y_train.shape, x_test.shape, y_test.shape
from sklearn.preprocessing import StandardScaler, MinMaxScaler



scaler = MinMaxScaler()

x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns)



scaler = MinMaxScaler()

x_test = pd.DataFrame(scaler.fit_transform(x_test), columns=x_test.columns)
def train_model(x_train, y_train):

    dropout = 0.2

    activation = 'relu'

    input_dim = x_train.shape[1]



    model = Sequential()

    model.add(Dense(32,input_dim=input_dim, kernel_initializer='normal', activation=activation))

#     model.add(BatchNormalization())

#     model.add(Dropout(rate=dropout))

    model.add(Dense(32, kernel_initializer='normal', activation=activation))

#     model.add(BatchNormalization())

#     model.add(Dropout(rate=dropout))

    model.add(Dense(64, kernel_initializer='normal', activation=activation))

#     model.add(BatchNormalization())

#     model.add(Dropout(rate=dropout))

    model.add(Dense(64, kernel_initializer='normal', activation=activation))

#     model.add(BatchNormalization())

#     model.add(Dropout(rate=dropout))

#     model.add(Dense(40, activation=activation))

#     model.add(BatchNormalization())

#     model.add(Dropout(rate=dropout))

#     model.add(Dense(20, activation=activation))

#     model.add(BatchNormalization())

#     model.add(Dropout(rate=dropout))

    model.add(Dense(6,activation='softmax'))



    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_lab), y_lab)

#     class_weights = None



#     callbacks = [EarlyStopping(monitor='val_loss', patience=10),

#                  ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

    callbacks = None



    history=model.fit(x_train, y_train, validation_split=0.2, epochs=50, callbacks=callbacks, class_weight=class_weights, batch_size=8)

    

    return model, history
model, history = train_model(x_train, y_train)
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
# from keras.models import load_model

# model = load_model("best_model.h5")

 

predicted = model.predict(x_test)

predicted = np.argmax(predicted, axis=1)

pd.Series(predicted).value_counts()
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



print("Accuracy: ", accuracy_score(test_val, predicted))

    

cfm = confusion_matrix(test_val, predicted, labels = [0,1,2, 3, 4, 5])

print("Confusion Matrix: ")

print(cfm)



print("Classification Report: ")

print(classification_report(test_val, predicted))
model, _ = train_model(X, y)
# model = load_model("best_model.h5")
df_test = pd.read_csv("/kaggle/input/bitsf312-lab1/test.csv", index_col='ID')

df_test = pd.get_dummies(df_test)

df_test = df_test.drop(dropCols, axis = 1)



# scaler = MinMaxScaler()

# X_final = scaler.fit_transform(df_test)

X_final = df_test
final_pred = model.predict(X_final)

final_pred = np.argmax(final_pred, axis=1)

final_sub = pd.DataFrame({"ID":df_test.index, "Class":final_pred})

final_sub['Class'].value_counts()
final_sub.to_csv("sub1.csv", index=False)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df2, title = "Download CSV file", filename = "data.csv"):

    csv = df2.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(final_sub)