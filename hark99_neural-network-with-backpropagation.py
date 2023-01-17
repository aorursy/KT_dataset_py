# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/"))



# Any results you write to the current directory are saved as output.



# importing from keras



from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import adam

from sklearn.model_selection import train_test_split
# loading the digit dataset

df = np.loadtxt("../input/Attachment_1556072961 - Copy.csv")#, delimiter=",")

# split into input (X) and output (Y) variables

# split into input and output variables

X = df[:,1:256]

Y = df[:,0]

# seed for reproducing same results

seed = 20

np.random.seed(seed)



# split the data into training (80%) and testing (20%)

(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.20, random_state=seed)
# create the model

model = Sequential()

model.add(Dense(100, input_dim=255, init='uniform', activation='relu'))

model.add(Dense(100, init='uniform', activation='relu'))

model.add(Dense(1, init='uniform', activation='sigmoid'))



# compile the model

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



# fit the model

model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=5)#, verbose=0)
# evaluate the model

scores = model.evaluate(X_test, Y_test) #29.27

print("Accuracy: %.2f%%" % (scores[1]*100))
# predictions

predictions = model.predict(X_test)

# round predictions

rounded = [round(x[0]) for x in predictions]

print(rounded)