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
from sklearn.utils import shuffle
import keras
import tensorflow as tf

from sklearn import preprocessing
nd = pd.read_csv("../input/bearing-fault/nd.csv", names=["time", "acc1", "acc2", "acc3", "volt", "v0", "v1", "v2", "v3", "label"])
nd['category'] = 0
bf = pd.read_csv("../input/bearing-fault/bf.csv", names=["time", "acc1", "acc2", "acc3", "volt", "v0", "v1", "v2", "v3", "label"])
bf['category'] = 1
fur = pd.read_csv("../input/faults/fur.csv", names=["time", "acc1", "acc2", "acc3", "volt", "v0", "v1", "v2", "v3", "label"])
fur['category'] = 2
brf = pd.read_csv("../input/faults/brf.csv", names=["time", "acc1", "acc2", "acc3", "volt", "v0", "v1", "v2", "v3", "label"])
brf['category'] = 3
fbf = pd.read_csv("../input/faults/fbf.csv", names=["time", "acc1", "acc2", "acc3", "volt", "v0", "v1", "v2", "v3", "label"])
fbf['category'] = 4
frm = pd.read_csv("../input/faults/frm.csv", names=["time", "acc1", "acc2", "acc3", "volt", "v0", "v1", "v2", "v3", "label"])
frm['category'] = 5
fbrbf = pd.read_csv("../input/faults/fbrbf.csv", names=["time", "acc1", "acc2", "acc3", "volt", "v0", "v1", "v2", "v3", "label"])
fbrbf['category'] = 6


df = bf.append(nd)
df.columns = ["time", "acc1", "acc2", "acc3", "volt", "v0", "v1", "v2", "v3", "label","category"]
df = df.append(fur)
df.columns = ["time", "acc1", "acc2", "acc3", "volt", "v0", "v1", "v2", "v3", "label","category"]
df = df.append(brf)
df.columns = ["time", "acc1", "acc2", "acc3", "volt", "v0", "v1", "v2", "v3", "label","category"]
df = df.append(fbf)
df.columns = ["time", "acc1", "acc2", "acc3", "volt", "v0", "v1", "v2", "v3", "label","category"]
df = df.append(frm)
df.columns = ["time", "acc1", "acc2", "acc3", "volt", "v0", "v1", "v2", "v3", "label","category"]
df = df.append(fbrbf)
df.columns = ["time", "acc1", "acc2", "acc3", "volt", "v0", "v1", "v2", "v3", "label","category"]

df = df.drop(['label'], axis=1)
df.head()
tempx = df.iloc[:, 0:9]
tempy = df.iloc[:, 9:]

tempx.head()
tempy.head()
%matplotlib inline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter3D(df["acc1"], df["acc2"], df["acc3"], c=df["label"] ,cmap="Greens")
tempx = tempx.dropna()
dump = tempx.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(dump)
tempx = pd.DataFrame(x_scaled)

df = shuffle(df)

tempx.columns = ["time", "acc1", "acc2", "acc3", "volt", "v0", "v1", "v2", "v3"]
tempy.columns = ["category"]

df.iloc[:, 0:9] = tempx


df.head()


from sklearn.preprocessing import OneHotEncoder 
  
# creating one hot encoder object with categorical feature 0 
# indicating the first column 
onehotencoder = OneHotEncoder() 
Y = onehotencoder.fit_transform(df.category.values.reshape(-1,1)).toarray() 

dfOneHot = pd.DataFrame(Y)

print(dfOneHot.head())


X = df.iloc[:,0:9]




print(X.shape)
print(X.head())




print(Y.shape)
print(Y.head())
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = 0.4, random_state = 1,shuffle=True)


print(X_train.head())

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical

model = keras.Sequential([
    keras.layers.Dense(32,activation='relu',input_shape=(9,)),
    keras.layers.Dense(32,activation='relu'),
    keras.layers.Dense(16,activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(16,activation='relu'),
    keras.layers.Dense(8,activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(7, activation='softmax')
])
adam = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=10,batch_size=100,validation_split=0.2, shuffle='True' )
model.test_on_batch(X_test,y_test)
model.metrics_names
import matplotlib.pyplot as plt
print(history.history.keys())
#  "Accuracy"
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
import pandas as pd
bf = pd.read_csv("../input/bearing-fault/bf.csv")
nd = pd.read_csv("../input/bearing-fault/nd.csv")
import pandas as pd
brf = pd.read_csv("../input/brf.csv")
fbf = pd.read_csv("../input/fbf.csv")
fbrbf = pd.read_csv("../input/fbrbf.csv")
frm = pd.read_csv("../input/frm.csv")
fur = pd.read_csv("../input/fur.csv")