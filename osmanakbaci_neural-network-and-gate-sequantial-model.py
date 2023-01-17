# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
xs_and  = np.array([[0.0,0.0],[0.0,1.0],[1.0,0.0],[1.0,1.0]], dtype = float)
ys_and = np.array([0.0,0.0,0.0,1.0], dtype = float)
model_and = keras.Sequential()
model_and.add(keras.layers.Dense(units=2, input_shape=[2]))
model_and.add(keras.layers.Dense(units=1))

model_and.compile(optimizer="sgd", loss="mean_squared_error", metrics=["accuracy"])

model_and.fit(xs_and, ys_and, epochs=300, verbose=0)
model_and.predict(np.array([[1.0, 1.0]]))
first_layer_weights_and = model_and.layers[0].get_weights()[0]
first_layer_biases_and  = model_and.layers[0].get_weights()[1]
print("ilk katman ağırlıkları")
print(first_layer_weights_and)
print("\nilk katman yanlılıkları")
print(first_layer_biases_and)

second_layer_weights_and = model_and.layers[1].get_weights()[0]
second_layer_biases_and  = model_and.layers[1].get_weights()[1]
print("\nçıkış katmanı ağırlıkları")
print(second_layer_weights_and)
print("\nçıkış katmanı yanlılıkları")
print(second_layer_biases_and)