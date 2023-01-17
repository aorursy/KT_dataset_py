# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import pickle
import sklearn
from sklearn import linear_model

data = pd.read_csv("../input/AI_Final_Review.csv", sep=",")
print(data.head())
data['Housework'] = data.Housework.map({'Mot':1, 'Hai':2, 'Ba':3 , 'Bon':4 , 'Nam':5 , 'Sau':6 , 'Bay':7 , 'Tam':8, 'Chin':9, 'Muoi':10})
data['Discussion']= data.Discussion.fillna(data.Discussion.mean())
data['Housework']= data.Housework.fillna(data.Housework.mean())
data['Quizz']= data.Quizz.fillna(data.Quizz.mean())


x = data.iloc[:, 1:10].values
y = data.iloc[:, -1].values
print(x)
print(y)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.1,shuffle= False, stratify=None);
linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)
predictions = linear.predict([[6,7,8,7,9,9,8,5,8]])
print(predictions)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler as ss
sc = ss()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)




#Neural network
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("../input/AI_Final_Review.csv", sep=",")
print(data.head())
data['Housework'] = data.Housework.map({'Mot':1, 'Hai':2, 'Ba':3 , 'Bon':4 , 'Nam':5 , 'Sau':6 , 'Bay':7 , 'Tam':8, 'Chin':9, 'Muoi':10})
data['Discussion']= data.Discussion.fillna(data.Discussion.mean())
data['Housework']= data.Housework.fillna(data.Housework.mean())
data['Quizz']= data.Quizz.fillna(data.Quizz.mean())


x = data.iloc[:, 1:10].values
y = data.iloc[:, -1].values
print(x)
print(y)

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9','10']
max_acc=0
for i in range(10):
    x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(x, y, test_size=0.2)
    model_test = keras.Sequential([   
        keras.layers.Dense(11, input_dim = 9, activation='relu'),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(11, activation="softmax")
        ])

    model_test.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    model_test.fit(x_train,y_train,epochs=50)
    test_loss,test_acc=model_test.evaluate(x_test,y_test)
    if test_acc>max_acc:       
        max_acc=test_acc
print(max_acc)

predictions = model_test.predict([[6,7,8,7,9,9,8,5,8]])
predictions_tong = model_test.predict([[8, 9, 7, 7, 7, 8, 8, 8, 7]])

print(predictions_tong)
result=class_names[np.argmax(predictions)]
print(result)
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

#read in training data
train_df = pd.read_csv("../input/finalreview/AI_Final_Review.csv", sep=",")
train_df.head()
train_df['Housework'] = train_df.Housework.map({'Mot':1, 'Hai':2, 'Ba':3 , 'Bon':4 , 'Nam':5 , 'Sau':6 , 'Bay':7 , 'Tam':8, 'Chin':9, 'Muoi':10})
train_df['Discussion']= train_df.Discussion.fillna(train_df.Discussion.mean())
train_df['Housework']= train_df.Housework.fillna(train_df.Housework.mean())
train_df['Quizz']= train_df.Quizz.fillna(train_df.Quizz.mean())




target = 'Final Grade'

train_df = train_df.iloc[:, 1:11]


scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train = scaler.fit_transform(train_df)

# Print out the adjustment that the scaler applied to the total_earnings column of data
# print("Note: median values were scaled by multiplying by {:.10f} and adding {:.6f}".format(scaler.scale_[10], scaler.min_[10]))
multiplied_by = scaler.scale_[9]
added = scaler.min_[9]

scaled_train_df = pd.DataFrame(scaled_train, columns=train_df.columns.values)

#imports
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense

#build our model
model = Sequential()

model.add(Dense(50, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

X = scaled_train_df.drop(target, axis=1).values
Y = scaled_train_df[[target]].values

# Train the model
model.fit(
    X[10:],
    Y[10:],
    epochs=50,
    shuffle=True,
    verbose=2
)

#inference
prediction = model.predict(X[[8, 9, 7, 7, 7, 8, 8, 8, 7]])
y_0 = prediction[0][0]
print('Prediction with scaling - {}',format(y_0))
y_0 -= added
y_0 /= multiplied_by
print(" Prediction  - ${}".format(y_0))



#linear, logis, svm, #Neural network , 