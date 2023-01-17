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
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

path = '/kaggle/input/pima-diabetes-database/Pima Indians Diabetes Database.csv'
x = pd.read_csv(path)
y = x['Class variable (0 or 1)']
x = x.drop(columns = 'Class variable (0 or 1)')
y
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 42)
model = Sequential([
    Dense(64,input_dim = 8,activation='relu'),
    Dense(32,activation='relu'),
    Dense(1,activation='sigmoid')
])
model.compile(optimizer='sgd',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(np.array(x_train),np.array(y_train),epochs=100)
model.evaluate(np.array(x_test),np.array(y_test))
