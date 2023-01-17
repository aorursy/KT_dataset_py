import os

print(os.listdir("../input"))

import pickle



Xfile = open('../input/processeddata/trainPickle', 'rb')      

X = pickle.load(Xfile)

Xfile.close()



X_testfile = open('../input/processedtestdata/testPickle', 'rb')      

X_test = pickle.load(X_testfile)

X_testfile.close()

#len(X_test)

import pandas as pd

import numpy as np

import os

print(os.listdir("../input"))

datas = pd.read_csv("../input/jigsawunintendedbiasintoxicitytrain/train.csv")

#datas.head()
#select only required column

ndata = datas.loc[:,['id','comment_text','target']]



#ndata.head()
#preparing target label data

y_t = ndata.target



del datas, ndata

y = [1 if t>=0.5 else 0 for t in y_t]



#from previous kernels found values

avgWordInlst = 50

uniqueCount = 524535

#**********Creating LSTM model****************



from keras.models import Model, Input, Sequential

from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional



model = Sequential()

model.add(Embedding(input_dim=uniqueCount, output_dim=150, input_length=avgWordInlst))

model.add(Dropout(0.2))

model.add(LSTM(100))

model.add(Dropout(0.2))

model.add(Dense(1, kernel_initializer='normal',activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

model.fit(X, y,validation_split=0.05, epochs=2, batch_size=50,verbose=2)
tdata = pd.read_csv("../input/jigsawunintendedbiasintoxicityclassification/test.csv")
#Predicting and writing prediction to file

y_pred = model.predict(X_test)





dfSample = pd.DataFrame({"id":tdata["id"].values})



dfSample['prediction'] = y_pred

dfSample.to_csv("submission.csv", index=False)