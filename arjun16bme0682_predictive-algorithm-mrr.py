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
import pandas as pd

import numpy as np
df = pd.DataFrame({'Abrasive Percentage':[10,10,10,20,20,20,30,30,30],

                   'Pressure':[35000,40000,45000,35000,40000,45000,35000,40000,45000],

                   'Stand-Off Distance':[1,1.5,2,1.5,2,1,2,1,1.5],

                   'Traverse Speed':[52.19,63.52,75.77,52.19,63.52,75.77,52.19,63.52,75.77]

                  })
xt = pd.DataFrame({'Abrasive Percentage':[10,10,10,20,20,20,30,30,30],

                   'Pressure':[35000,40000,45000,35000,40000,45000,35000,40000,45000],

                   'Stand-Off Distance':[1,1.5,2,1.5,2,1,2,1,1.5],

                   'Traverse Speed':[52.19,63.52,75.77,52.19,63.52,75.77,52.19,63.52,75.77]

                  })
yt=np.array([494.337,604.712,715.381,532.932,619.071,707.475,491.088,609.172,733.014])
import tensorflow as tf

from tensorflow import keras

model=tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[4])])

rmsprop=keras.optimizers.RMSprop(lr=0.0001, rho=0.9)

model.compile(optimizer=rmsprop, loss='mean_squared_error')

xs=df

ys=np.array([494.337,604.712,715.381,532.932,619.071,707.475,491.088,609.172,733.014])

model.fit(xs, ys, epochs=100000)
model.evaluate(xt,yt)
for x in [10,20,30]:

    for y in [35000,40000,45000]:

        for z in [1,1.5,2]:

            for p in [52.19,63.52,75.77]:

                check=pd.DataFrame({'Abrasive Percentage':[x],

                   'Pressure':[y],

                   'Stand-Off Distance':[z],

                   'Traverse Speed':[p]

                  })

                q=model.predict(check)

                print(x,y,z,p,q)