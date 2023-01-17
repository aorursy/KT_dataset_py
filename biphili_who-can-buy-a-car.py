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
import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt 

import seaborn as sns
df=pd.read_csv('../input/car-purchase-data/Car_Purchasing_Data.csv',encoding='ISO-8859-1')
df.head()
sns.pairplot(df)

plt.ioff()
X=df.drop(['Customer Name','Customer e-mail','Country','Car Purchase Amount'],axis=1)
#X
y=df['Car Purchase Amount']
#y
X.shape
y.shape
from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()

X_scaled=scaler.fit_transform(X)
X_scaled
X_scaled.shape
scaler.data_max_
scaler.data_min_
y=y.values.reshape(-1,1)
y.shape
y_scaled=scaler.fit_transform(y)
#y_scaled
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X_scaled,y_scaled,test_size=0.15)
X_train.shape
X_test.shape
import tensorflow.keras 

from keras.models import Sequential 

from keras.layers import Dense 



model=Sequential()

model.add(Dense(40,input_dim=5,activation='relu'))

model.add(Dense(40,activation='relu'))

model.add(Dense(1,activation='linear'))

model.summary()
model.compile(optimizer='adam',loss='mean_squared_error')
epochs_hist=model.fit(X_train,y_train,epochs=100,batch_size=50,verbose=1,validation_split=0.2)
epochs_hist.history.keys()
plt.plot(epochs_hist.history['loss'])

plt.plot(epochs_hist.history['val_loss'])

plt.xlabel('Epoch Number')

plt.ylabel('Training and Validation Loss')

plt.title('Model Loss Progress during training')

plt.legend(['Training Loss','Validation Loss'])

plt.ioff()
# Gender,Age,Annual Salary,Credit card debt,Net Worth 

X_test=np.array([[1,50,50000,10000,600000]])

y_predict=model.predict(X_test)
print('Expected Purchase Amount',y_predict)