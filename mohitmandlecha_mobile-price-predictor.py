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
dataset=pd.read_csv('/kaggle/input/mobile-price-classification/train.csv')

dataset.price_range.unique
dataset.head(100)
X=dataset.iloc[:,:-1].values

y=dataset.iloc[:,20:21].values
X
dataset.describe()
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X = sc.fit_transform(X)#Normalizing the data









from sklearn.preprocessing import OneHotEncoder

ohe=OneHotEncoder()

y=ohe.fit_transform(y).toarray()



from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1)

y_train.shape
import keras

from keras.models import Sequential

from keras.layers import Dense,Dropout



def mymodel():



    model=Sequential()

    model.add(Dense(20,input_dim=20,activation='relu'))

    model.add(Dense(16,activation='relu'))

    model.add(Dense(12,activation='relu'))

    model.add(Dense(6,activation='relu'))

    model.add(Dropout(0.2))

    model.add(Dense(4,activation='softmax'))

    

    return model
model=mymodel()

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
History = model.fit(X_train, y_train, epochs=100, batch_size=64)
history = model.fit(X_train, y_train,validation_data = (X_test,y_test), epochs=100, batch_size=64)
 

y_pred = model.predict(X_test)

#Converting predictions to label

pred = list()

for i in range(len(y_pred)):

    pred.append(np.argmax(y_pred[i]))

#Converting one hot encoded test label to label

test = list()

for i in range(len(y_test)):

    test.append(np.argmax(y_test[i]))

    

from sklearn.metrics import accuracy_score

a = accuracy_score(pred,test)

print('Accuracy is:', a*100)

