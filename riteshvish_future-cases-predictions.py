import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from tensorflow.keras.layers import LSTM, GRU, Dense, Flatten,Dropout

from tensorflow.keras.models import Sequential
df = pd.read_csv("../input/covid19-in-india/covid_19_india.csv")

df.shape
df.head()
dff=df.loc[:,["Date","Confirmed"]]

dff
a=dff.groupby(['Date']).sum()

a


a["date"]=a.index

a
b=a.reset_index(drop=True)



b
b["Date"] = pd.to_datetime(b["date"],dayfirst=True)

b
b = b.sort_values(by="Date")

b=b.reset_index(drop=True)

b
b=b.Confirmed.values

b
def split_sequence(sequence, n_steps):

	X, y = list(), list()

	for i in range(len(sequence)):

		# find the end of this pattern

		end_ix = i + n_steps

		# check if we are beyond the sequence

		if end_ix > len(sequence)-1:

			break

		# gather input and output parts of the pattern

		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]

		X.append(seq_x)

		y.append(seq_y)

	return np.array(X), np.array(y)
X,y=split_sequence(b,10)

print(X[-1])

print(y[-1])
n_steps=10

n_features=1

model=Sequential()

model.add(LSTM(300,activation='relu',input_shape=(n_steps,n_features)))



model.add(Dense(1))

model.compile("adam","mse")

model.fit(X.reshape(X.shape[0],n_steps,n_features),y,epochs=200)

# test_sample=np.array([[4023179, 4113811, 4204613, 4280422, 4370128, 4465863, 4562414]]).reshape(1,n_steps,n_features)

# model.predict(test_sample)
test_sample=np.array([[3936747 ,4023179 ,4113811 ,4204613 ,4280422,44370128 ,4465863 ,4562414, 4659984,4754356]]).reshape(1,n_steps,n_features)

model.predict(test_sample)
n_steps=7

n_features=1

model=Sequential()

model.add(GRU(300,activation='relu',input_shape=(n_steps,n_features)))

model.add(Dense(1))

model.compile("adam","mse")

model.fit(X.reshape(X.shape[0],n_steps,n_features),y,epochs=300,verbose=0)

test_sample=np.array([[3936747 ,4023179 ,4113811 ,4204613 ,4280422,44370128 ,4465863 ,4562414, 4659984,4754356]]).reshape(1,n_steps,n_features)

model.predict(test_sample)