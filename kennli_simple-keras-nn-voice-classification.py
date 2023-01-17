# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import time 

from keras.models import Sequential

from keras.layers import Dense



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/voice.csv")

data.head()
#Clean labels

data.loc[data['label']=='male', 'label'] = 0

data.loc[data['label']=='female', 'label'] = 1
#Split data into training/test sets by a 80/20 split

train=data.sample(frac=0.8,random_state=200)

test=data.drop(train.index)
print(data.shape)

np.random.seed(1)

X_train = np.array(train.iloc[:,0:20])

Y_train = np.array(train.iloc[:,20])

X_test = np.array(test.iloc[:,0:20])

Y_test = np.array(test.iloc[:,0:20])
model = Sequential()

model.add(Dense(4, input_dim=20, activation='relu'))

model.add(Dense(2, activation='relu'))

model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
start = time.time()

model.fit(X_train,Y_train, epochs=150, batch_size=10, verbose=0)

print("seconds to fit:", time.time()-start)
predictions = model.predict(X_test)

rounded = [round(x[0]) for x in predictions]

print(rounded[:40])
#Calculate accuracy of predictions

diff = np.array(test['label'].values) - rounded



values, counts = np.unique(1,return_counts=True )

print("Accuracy: ",1-(counts/len(diff)))