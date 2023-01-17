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
import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

df = pd.read_csv('../input/housing-data.csv')

df.head()
plt.figure(figsize = (15,5))

for i, feature in enumerate(df.columns):

    plt.subplot(1,4,i+1)

    df[feature].plot(kind='hist',title =feature)

    plt.xlabel(feature)
X = df[['sqft','bdrms','age']].values

y = df[['price']].values
X = df[['sqft','bdrms','age']].values

y = df['price'].values
from keras.models import Sequential 

from keras.layers import Dense 

from keras.optimizers import Adam
model = Sequential()

model.add(Dense(1,input_shape=(3,)))

model.compile(Adam(lr=0.8),'mean_squared_error')
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)

len(X_train)
model.fit(X_train,y_train,epochs=10)
df['price'].min()
df['price'].max()
df.describe()
df['sqft1000'] = df['sqft']/1000

df['age10'] = df['age']/10

df['price100k'] = df['price']/1e5
X = df[['sqft1000','bdrms','age10']].values

y = df['price100k'].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
model = Sequential()

model.add(Dense(1,input_dim =3))

model.compile(Adam(lr=0.1),'mean_squared_error')

model.fit(X_train,y_train,epochs = 50)
from sklearn.metrics import r2_score
y_train_pred = model.predict(X_train)

y_test_pred = model.predict(X_test)



print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train,y_train_pred)))

print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test,y_test_pred)))