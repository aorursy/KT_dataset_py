from keras.models import Sequential
from keras.layers import Dense 
from keras.optimizers import Adam, SGD

%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/weight-height.csv')
X=df[['Height']].values
y_true=df[['Weight']].values
X
model = Sequential()
model.add(Dense(1, input_shape=(1,)))
model.summary()
model.compile(Adam(lr=0.8), 'mean_squared_error')
model.fit(X,y_true, epochs=35, batch_size=110)
y_pred= model.predict(X)
df.plot(kind='scatter',
       x='Height',
       y='Weight', title='Weight and Height in adults')
plt.plot(X, y_pred, color='red', linewidth=3)
w,b=model.get_weights()
w
b
