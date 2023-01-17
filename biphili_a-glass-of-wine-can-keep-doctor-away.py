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
df = pd.read_csv('../input/principal-component-analysis/Wine.csv')

df.head()
y = df['Customer_Segment']
y.value_counts()
y_cat = pd.get_dummies(y)
y_cat.head()
X = df.drop('Customer_Segment',axis=1)
X.shape
import seaborn as sns
sns.pairplot(df,hue = 'Customer_Segment');
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

Xsc = sc.fit_transform(X)
from keras.models import Sequential 

from keras.layers import Dense

from keras.optimizers import SGD, Adam, Adadelta, RMSprop

import keras.backend as K
K.clear_session()

model = Sequential()

model.add(Dense(5,input_shape=(13,),kernel_initializer = 'he_normal',activation = 'relu'))

model.add(Dense(3,activation = 'softmax'))



model.compile(RMSprop(lr=0.1),'categorical_crossentropy',metrics = ['accuracy'])



model.fit(Xsc,y_cat.values,batch_size=8,epochs=10,verbose=1,validation_split=0.2)
K.clear_session()

model = Sequential()

model.add(Dense(8,input_shape=(13,),kernel_initializer = 'he_normal',activation = 'tanh'))

model.add(Dense(5,kernel_initializer = 'he_normal',activation = 'tanh'))

model.add(Dense(2,kernel_initializer = 'he_normal',activation = 'tanh'))

model.add(Dense(3,activation = 'softmax'))





model.compile(RMSprop(lr=0.05),'categorical_crossentropy',metrics = ['accuracy'])



model.fit(Xsc,y_cat.values,batch_size=16,epochs=20,verbose=1)
model.summary()
inp = model.layers[0].input

out = model.layers[2].output
features_function = K.function([inp],[out])
features = features_function([Xsc])[0]
plt.scatter(features[:,0],features[:, 1])