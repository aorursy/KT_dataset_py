# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

from numpy import loadtxt 

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Sequential 

from keras.layers import Dense

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data =  pd.read_csv("/kaggle/input/the-wisconsin-cancer-dataset/breast-cancer-wisconsin.data", header=None)

data.dtypes
data2 = [x_trn, y_trn]

big_table = pd.concat(data2, axis = 1)

corr_matrix = big_table.corr()

sns.heatmap(corr_matrix, annot = True) 
x_trn = data[[1,2,3,4,5,7,8,9]]

y_trn = data[10]//4

x_tst = data[[1,2,3,4,5,7,8,9]]



y_trn 
model = Sequential()

model.add(Dense(12, activation='relu'))

model.add(Dense(8,activation='relu'))

model.add(Dense(5,activation='relu'))

model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_trn, y_trn, epochs=50, batch_size=20)

predictions = model.predict_classes(x_tst)