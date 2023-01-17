# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from keras.models import Sequential

from keras.layers import Dense

from sklearn import model_selection



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/FullData.csv')
dx, dy = df['Ball_Control'], df['Age']
x, x_val, y, y_val = model_selection.train_test_split(dx, dy, test_size=0.2, random_state=42)
x = np.array([v for v in x if v >= 0])
x_val = np.array([v for v in x_val if v >= 0])
y = np.array([v for v in y if v >= 0])
y_val = np.array([v for v in y_val if v >= 0])
model = Sequential()
model.add(Dense(1, input_dim=1))
model.compile(optimizer='rmsprop', loss='mse')
model.fit(x, y, epochs=10, validation_data=(x_val, y_val))
model = Sequential()
model.add(Dense(1, activation='sigmoid', input_dim=1))
model.compile(optimizer='rmsprop', loss='binary_crossentropy')
model.fit(x, y, epochs=10, validation_data=(x_val, y_val))
from keras.regularizers import L1L2 as l1l2
reg = l1l2(l1=0.01, l2=0.01)
model = Sequential()
model.add(Dense(1, activation='sigmoid', W_regularizer=reg, input_dim=1))
model.compile(optimizer='rmsprop', loss='binary_crossentropy')