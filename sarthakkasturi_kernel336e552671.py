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
from keras.models import Sequential
from keras.layers import Dense

from keras.layers import Dropout

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.preprocessing import StandardScaler
dataset=pd.read_csv("/kaggle/input/sonar (2).csv")
dataset.groupby("R/M").size()
dataset.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1, figsize=(12,12))

plt.show()
dataset.plot(kind='density', subplots=True, layout=(8,8), sharex=False, legend=False, fontsize=1, figsize=(12,12))

plt.show()

fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(dataset.corr(), vmin=-1, vmax=1, interpolation='none')

fig.colorbar(cax)

fig.set_size_inches(10,10)

plt.show()
array=dataset.values

X = array[:,0:-1].astype(float)

Y = array[:,-1]

scaler=StandardScaler()

X=scaler.fit_transform(X)

encoder= LabelEncoder()

Y=encoder.fit_transform(Y)

validation_size = 0.2

seed = 7

X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)




model=Sequential()

model.add(Dense(32,input_dim=60,activation='relu',kernel_initializer='uniform'))

model.add(Dropout(rate=0.1))

model.add(Dense(18,activation='relu',kernel_initializer='uniform'))

model.add(Dropout(rate=0.05))

model.add(Dense(9,activation='relu',kernel_initializer='uniform'))

model.add(Dense(1,activation='sigmoid',kernel_initializer='uniform'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(X_train,Y_train,epochs=500,batch_size=512)
Y_pred=model.predict(X_validation)>0.6
cm=confusion_matrix(Y_validation,Y_pred)
print(classification_report(Y_validation,Y_pred))

cm