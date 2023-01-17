# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/housing.csv")
import warnings

warnings.filterwarnings('ignore')
data.head()

print(data.shape)
data.isnull().sum()

data.info()
data.fillna(data['total_bedrooms'].mean(), inplace=True)
data.head()
data["ocean_proximity"].value_counts()
from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer



df = data.drop(labels=['median_house_value'], axis=1)

num_attrs = list(df)

num_attrs.remove("ocean_proximity")

cat_attrs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([

    ("num", SimpleImputer(strategy='median'),num_attrs),

    ("cat", OneHotEncoder(), cat_attrs),

])

X = full_pipeline.fit_transform(df)

X = pd.DataFrame(X, columns = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 

                               'population', 'households', 'median_income', '<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'NEAR BAY', 'ISLAND'])

X.head()
#train test split

from sklearn.model_selection import train_test_split



y = data.iloc[:, 9:]

X_train, X_remain, y_train, y_remain = train_test_split(X, y, test_size=0.35, random_state=1105233)

print(X_train.shape, y_train.shape)

print(X_remain.shape, y_remain.shape)
X_test, X_val, y_test, y_val = train_test_split(X_remain, y_remain, test_size=0.42857, random_state=1105233)

print(X_val.shape, y_val.shape)

print(X_test.shape, y_test.shape)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_val_scaled = scaler.transform(X_val)

X_test_scaled = scaler.transform(X_test)

print(X_train_scaled.shape, y_train.shape)
import keras

from keras.models import Sequential

from keras.layers import Dense


model = Sequential()

model.add(Dense(300, input_shape=X_train_scaled.shape[1:]))

model.add(Dense(300,activation="relu"))

model.add(Dense(300,activation="relu"))

model.add(Dense(300,activation="relu"))

model.add(Dense(300,activation="relu"))

model.add(Dense(300,activation="relu"))

model.add(Dense(300,activation="relu"))

model.add(Dense(300,activation="relu"))

model.add(Dense(300,activation="relu"))

model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=(["accuracy", 'mae']))
history=model.fit(X_train_scaled, y_train.values, epochs=30, 

          validation_data=(X_val_scaled, y_val))
model.summary()
import matplotlib.pyplot as plt

plt.plot(history.history['mean_absolute_error'], color = 'green')

plt.plot(history.history['val_mean_absolute_error'])

plt.show()
pred = model.predict(X_test_scaled)
from sklearn.metrics import mean_absolute_error

score = mean_absolute_error(y_test, pred)

print(score)
total=0

for ex in range(0, len(pred)):

    if(pred[ex] > 500000 or pred[ex] < 15000):

        total+=1

print(total)

        
from keras.datasets import cifar10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print(X_train.shape)

from keras.utils import to_categorical

y_train=to_categorical(y=y_train, num_classes=10)

y_test=to_categorical(y=y_test, num_classes=10)
X_train= X_train/255

X_test= X_test/255
from functools import partial



DefaultConv2D = partial(keras.layers.Conv2D,

                        kernel_size=3, activation='relu', padding ="SAME")

model=Sequential([

    DefaultConv2D(filters=64, kernel_size=7, input_shape=[32,32,3]),

    keras.layers.MaxPooling2D(pool_size=2),

    DefaultConv2D(filters=128),

    DefaultConv2D(filters=128),

    keras.layers.MaxPooling2D(pool_size=2),

    DefaultConv2D(filters=256),

    DefaultConv2D(filters=256),

    keras.layers.MaxPooling2D(pool_size=2),

    keras.layers.Flatten(),

    keras.layers.Dense(units=128, activation='relu'),

    keras.layers.Dropout(0.5),

    keras.layers.Dense(units=64, activation='relu'),

    keras.layers.Dropout(0.5),

    keras.layers.Dense(units=10, activation='softmax')

])

from keras import optimizers

optimizer= optimizers.SGD(lr=0.048)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=30, validation_split=0.15)
model.summary()

ev=model.evaluate(X_test, y_test)
ev
pred=model.predict(X_test)
print(pred[9])
result = pd.DataFrame(columns= ['ID', 'Label'])



classification = []

for i in range(0, len(pred)):

    classification.append(np.argmax(pred[i]))
miss_class= []

for i in range(10000):

    if(classification[i] != np.argmax(y_test[i])):

        miss_class.append([np.argmax(y_test[i]), classification[i], (pred[i, classification[i]]-pred[i, np.argmax(y_test[i])]), i])
print(len(miss_class))
table = pd.DataFrame(miss_class)

table.columns = ['Truth', 'Predicted_as', 'Error', 'Index']
table
worst0 = 0

worst1 = 0

worst2 = 0

worst3 = 0

worst4 = 0

worst5 = 0

worst6 = 0

worst7 = 0

worst8 = 0

worst9 = 0
for i in range(len(table)):

    if(table.iat[i, 0] == 0):

        if(table.iat[i, 2]>worst0):

            worst0=table.iat[i,2]

            index_worst0 = i

    if(table.iat[i, 0] == 1):

        if(table.iat[i, 2]>worst1):

            worst1=table.iat[i,2]

            index_worst1 = i

    if(table.iat[i, 0] == 2):

        if(table.iat[i, 2]>worst2):

            worst2=table.iat[i,2]

            index_worst2 = i

    if(table.iat[i, 0] == 3):

        if(table.iat[i, 2]>worst3):

            worst3=table.iat[i,2]

            index_worst3 = i

    if(table.iat[i, 0] == 4):

        if(table.iat[i, 2]>worst4):

            worst4=table.iat[i,2]

            index_worst4 = i

    if(table.iat[i, 0] == 5):

        if(table.iat[i, 2]>worst5):

            worst5=table.iat[i,2]

            index_worst5= i

    if(table.iat[i, 0] == 6):

        if(table.iat[i, 2]>worst6):

            worst6=table.iat[i,2]

            index_worst6 = i

    if(table.iat[i, 0] == 7):

        if(table.iat[i, 2]>worst7):

            worst7=table.iat[i,2]

            index_worst7 = i

    if(table.iat[i, 0] == 8):

        if(table.iat[i, 2]>worst8):

            worst8=table.iat[i,2]

            index_worst8 = i

    if(table.iat[i, 0] == 9):

        if(table.iat[i, 2]>worst9):

            worst9=table.iat[i,2]

            index_worst9 = i
worst_missclassified = []

worst_missclassified.append(index_worst0)

worst_missclassified.append(index_worst1)

worst_missclassified.append(index_worst2)

worst_missclassified.append(index_worst3)

worst_missclassified.append(index_worst4)

worst_missclassified.append(index_worst5)

worst_missclassified.append(index_worst6)

worst_missclassified.append(index_worst7)

worst_missclassified.append(index_worst8)

worst_missclassified.append(index_worst9)

print(worst_missclassified)
for i in (worst_missclassified):

    #print(table.iat[i, 0], table.iat[i, 1], table.iat[i, 2])

    plt.imshow(X_test[table.iat[i, 3]])

    plt.title("Truth="+ str(table.iat[i, 0]) +", Predicted=" + str(table.iat[i, 1]) + ", P_predicted-P_correct=" + str(table.iat[i, 2]))

    plt.show()