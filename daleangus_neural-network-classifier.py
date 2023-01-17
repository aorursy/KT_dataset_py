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
data = pd.read_csv("/kaggle/input/magic-gamma-telescope-dataset/telescope_data.csv", sep=",")

column_names = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist']

label_name = ['class']
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder

X = StandardScaler().fit_transform(np.array(data[column_names]))

y = np.resize(np.array(LabelEncoder().fit_transform(data[label_name])), (X.shape[0], 1))

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

from keras.layers import Dense, Dropout

from keras.models import Sequential

from keras.optimizers import Adamax

model = Sequential()

model.add(Dense(128, activation='relu', input_dim=X_train.shape[1], kernel_initializer='he_normal'))

model.add(Dropout(0.5))

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(y_train.shape[1], activation='sigmoid'))



# opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# opt = SGD(lr=0.001, decay=1e-6)

opt = Adamax(lr=0.005, beta_1=0.9, beta_2=0.999)

# opt = RMSprop(lr=.002, rho=0.9)

model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])



history = model.fit(X_train, y_train, epochs=200, batch_size=32, verbose=0, validation_split=.3)

#predictions = model.predict(X_test)

score = model.evaluate(X_test, y_test, batch_size=1)

print(score)