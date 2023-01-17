from keras.models import Sequential
from keras.layers import Dense 
from keras.optimizers import Adam, SGD

%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))
file=pd.read_csv('../input/weight-height.csv')
X=file[['Height']].values
Y=file[['Weight']].values
X
Y
model = Sequential()
model.add(Dense(1, input_shape=(1,)))
model.summary()
model.compile(Adam(lr=0.8), 'mean_squared_error')
model.fit(X,Y, epochs=40, batch_size=120)
y_pred= model.predict(X)
file.plot(kind='scatter',
       x='Height',
       y='Weight', title='Weight and Height in adults')
plt.plot(X, y_pred, color='red', linewidth=3)
a,b=model.get_weights()
a
b
