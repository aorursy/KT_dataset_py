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
import pandas as pd

import numpy as np

data = np.load('../input/X_train.npy')

labels = np.load('../input/y_train.npy')

print(labels.shape,data.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25)
X_train = X_train.reshape(X_train.shape[0],-1)

X_test = X_test.reshape(X_test.shape[0],-1)

print(X_train.shape)

print(y_train[:10])
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(solver='saga',multi_class ='multinomial')

clf.fit(X_train, y_train)
def evaluate(pred,labels):

    return sum(pred == labels)/len(pred)
pred = clf.predict(X_test)

print("Accuracy:",evaluate(pred,y_test))
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_jobs=-1, n_estimators=10)

rfc.fit(X_train, y_train)
pred = rfc.predict(X_test)

print("Accuracy:",evaluate(pred,y_test))
from keras.models import Sequential

from keras.layers import Dense, Dropout



def cat_to_onehot(labels):

    labels_onehot = np.zeros((len(labels),10),dtype=np.int)

    for i in range(len(labels)):

        labels_onehot[i][labels[i]] = 1

    return labels_onehot



def onehot_to_cat(labels_onehot):

    labels = labels_onehot.argmax(axis=1)

    return labels



X_train = X_train/255

X_test = X_test/255



model = Sequential()

model.add(Dense(32, activation='relu', input_dim=(28*28)))

model.add(Dense(16, activation='relu'))

model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

print(model.summary())



y_train_onehot = cat_to_onehot(y_train)

y_test_onehot = cat_to_onehot(y_test)

print(X_train.shape,y_train.shape)

model.fit(X_train,y_train_onehot,epochs=10)
pred_onehot = model.predict(X_test)

evaluate(y_test,onehot_to_cat(pred_onehot))
unknown = np.load("../input/X_test.npy")

unknown = unknown.reshape(unknown.shape[0],-1)



pred = rfc.predict(unknown)
out = pd.DataFrame(columns=['Id','Prediction'])

out['Id'] = range(len(pred))

out['Prediction'] = pred

print(out.info())

out.head(10)
out.to_csv('./out.csv',index=False)