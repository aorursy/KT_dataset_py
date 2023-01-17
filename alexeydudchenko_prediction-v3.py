# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

np.random.seed(1337)  # for reproducibility

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from keras.models import Sequential

from keras.layers import Dense

import matplotlib.pyplot as plt 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import os

print(os.listdir("../input"))
data = pd.read_csv("../input/lkdkjgbvsxcs/ex3.csv", delimiter = ";")

prediction = pd.read_csv("../input/topredict/Prediction.csv", delimiter = ";", index_col=0)



data = data.iloc[:,:-6]

data = data.apply(lambda x: x.str.replace(',','.'))

prediction = prediction.apply(lambda x: x.str.replace(',','.'))



data.head()
data1 = data.iloc[:,:-1]

data2_x = data.iloc[:,:-2] 

data2_y = data.iloc[:,-1:]

data1_y = data.iloc[:,-2:-1]

data1_x = data.iloc[:,:-2]
from sklearn.model_selection import train_test_split

X2_train, X2_test, Y2_train, Y2_test = train_test_split(data2_x, data2_y, test_size = 0.23, random_state = 0)

X1_train, X1_test, Y1_train, Y1_test = train_test_split(data1_x, data1_y, test_size = 0.23, random_state = 0)
X2_train
model2 = Sequential()

model2.add(Dense(output_dim=1, input_dim=4))

model2.compile(loss='mse', optimizer='sgd')



print('Training Model 2 -----------')

for step in range(301):

    cost = model2.train_on_batch(X2_train, Y2_train)

    if step % 100 == 0:

        print('train cost: ', cost)
from keras.models import Sequential

from keras.optimizers import Adam

from keras.layers.normalization import BatchNormalization

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.layers.core import Activation

from keras.layers.core import Dropout

from keras.layers.core import Dense

from keras.layers import Flatten

from keras.layers import Input

from keras.models import Model

 

def create_mlp(dim, regress=False):

	# define our MLP network

	model = Sequential()

	model.add(Dense(8, input_dim=dim, activation="relu"))

	model.add(Dense(4, activation="relu"))

 

	# check to see if the regression node should be added

	if regress:

		model.add(Dense(1, activation="linear"))

 

	# return our model

	return model
model2_2 = create_mlp(4, regress=True)

opt = Adam(lr=1e-3, decay=1e-3 / 200)

model2_2.compile(optimizer="Nadam", loss="mean_squared_error", metrics=["mean_squared_error"])

 

# train the model

print("[INFO] training model...")

model2_2.fit(X2_train, Y2_train, validation_data=(X2_test, Y2_test),

	epochs=200, batch_size=8)

print(model2_2.evaluate(X2_test, Y2_test))
model1_2 = create_mlp(4, regress=True)

opt = Adam(lr=1e-3, decay=1e-3 / 200)

model1_2.compile(optimizer="Nadam", loss="mean_squared_error", metrics=["mean_squared_error"])

 

# train the model

print("[INFO] training model...")

model1_2.fit(X1_train, Y1_train, validation_data=(X1_test, Y1_test),

	epochs=200, batch_size=8)

print(model1_2.evaluate(X1_test, Y1_test))
print("Model 1_2" , model1_2.evaluate(X1_test, Y1_test))
print("Model 2_2" , model2_2.evaluate(X2_test, Y2_test))
X2_train
predicted1 = model2_2.predict(prediction)

predicted1
predicted2 = model1_2.predict(prediction)



predicted2
predicted1DF = pd.DataFrame(data=predicted1)

predicted1DF.to_csv('mycsvfile1.csv',index=False, decimal=',')

predicted2DF = pd.DataFrame(data=predicted2)

predicted2DF.to_csv('mycsvfile2.csv',index=False, decimal=',')