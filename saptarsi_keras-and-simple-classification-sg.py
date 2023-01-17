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
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split
df=pd.read_csv('/kaggle/input/breastcancer/breastcancer.csv')
df.head(3)
X=df.iloc[:,1:10]
y=df.iloc[:,10]
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)
from sklearn import preprocessing
X_scaled = preprocessing.scale(X)
X_train, X_test, y_train, y_test = train_test_split( X_scaled, encoded_y, test_size=0.2, random_state=12)
model = keras.models.Sequential()
model.add(keras.layers.Dense(20,input_dim=9, activation="relu"))
model.add(keras.layers.Dense(8, activation="relu"))
#model.add(keras.layers.Dense(25, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))
from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
model.compile(loss="binary_crossentropy",optimizer="sgd",metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=100)
import pandas as pd
import matplotlib.pyplot as plt
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
plt.show()
model.summary()
model.evaluate(X_test, y_test)