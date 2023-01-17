def function(x):

     y = (2 * x) + (x - 1)

     return y

function(18.0)
import tensorflow as tf

import numpy as np

from tensorflow import keras



model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

model.compile(optimizer='sgd', loss='mean_squared_error')

x = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)

y = np.array([-4.0, -1.0, 2.0, 5.0, 8.0, 11.0,14.0 ], dtype=float)

model.fit(x, y, epochs=500)
model.predict([18.0])
import numpy as np

from sklearn.linear_model import LinearRegression



x = np.array([[4642146.0, 3519515.0, 1728786.0, 1708590.0, 1534749.0]], dtype=float)

y = np.array([[10062.83, 7407.06, 3815.79, 3960.27, 3486.05]], dtype=float)

n = np.array([[5642146.0, 4519515.0, 2728786.0, 2708590.0, 2534749.0]], dtype=float)



reg = LinearRegression()
reg.fit(x, y)

reg.predict([[45.0]])