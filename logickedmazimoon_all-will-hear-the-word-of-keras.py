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
import tensorflow as tf

from keras.models import Sequential

import pandas as pd

from keras.layers import Dense



url = "https://raw.githubusercontent.com/Vlad0f/OHE-bgg-Reviews/master/OHE'dtable.csv"



data = pd.read_csv(url, delimiter=',')



labels=data['Target']

features = data.iloc[:,0:333332]
import numpy as np

from sklearn.model_selection import train_test_split



X=features



y=np.ravel(labels)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler().fit(X_train)



X_train = scaler.transform(X_train)



X_test = scaler.transform(X_test)
model = Sequential()



model.add(Dense(8, activation='relu', input_shape=(333332,)))



model.add(Dense(8, activation='relu'))



model.add(Dense(1, activation='sigmoid'))





model.compile(loss='binary_crossentropy',

              optimizer='sgd',

              metrics=['accuracy'])

                   

model.fit(X_train, y_train,epochs=8, batch_size=1, verbose=1)
y_pred = model.predict(X_test)



score = model.evaluate(X_test, y_test,verbose=1)



print(score)