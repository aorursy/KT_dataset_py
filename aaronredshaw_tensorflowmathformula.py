# By Aaron Redshaw



import tensorflow as tf

import numpy as np

from tensorflow import keras



model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])



model.compile(optimizer='sgd', loss='mse')



xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float) # see formula on next line.

ys = np.array([-1.0, 0.5, 2.0, 3.5, 5.0, 6.5], dtype=float) # using the formula y = (3x + 1)/2.



model.fit(xs, ys, epochs=5)

print(model.predict([7.0])) # Only 5 iterations. This won't be very accurate. You will see this on this page.

print("\n")



model.fit(xs, ys, epochs=20)

print(model.predict([7.0])) # After 20 iterations, it should be better, but not really accurate. Scroll down to 20 to see this.

print("\n")



model.fit(xs, ys, epochs=500)

print(model.predict([7.0])) # 500 iterations. Getting there. Scroll down to 500 to see this.

print("\n")



model.fit(xs, ys, epochs=1000)

print(model.predict([7.0])) # 1000 iterations. Very accurate. Never perfect, however. Scroll down to the bottom to see this.



# y = (3x + 1)/2. So, 3x7=21, 21+1=22, 22/2=11

# Your output should be something aproaching 11