# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
input_data = pd.read_csv("/kaggle/input/nasa-airfoil-self-noise/NASA_airfoil_self_noise.csv")
input_data.describe()
input_data.corr()
input_data["Frequency"].plot(kind="hist")
input_data["AngleAttack"].plot(kind="hist")
input_data["ChordLength"].plot(kind="hist")
input_data["FreeStreamVelocity"].plot(kind="hist")
input_data["SuctionSide"].plot(kind="hist")
input_data["Sound"].plot(kind="hist")
y = input_data["Sound"]
X = input_data.drop("Sound", axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)
import keras
from keras.layers import *
model = keras.models.Sequential()
model.add(Dense(128, input_dim = 5, activation="relu"))
model.add(Dense(256, activation="relu"))
model.add(Dense(256, activation="relu"))
model.add(Dense(256, activation="relu"))
model.add(Dense(256, activation="relu"))
model.add(Dense(256, activation="relu"))
model.add(Dense(256, activation="relu"))
model.add(Dense(256, activation="relu"))
model.add(Dense(512, activation="relu"))
model.add(Dense(1, activation="linear"))
from keras import metrics
model.compile(optimizer="adam", loss = "mean_squared_error", metrics=[metrics.MeanSquaredError()])
model.fit(X_train, y_train, batch_size = 64, epochs = 5000, verbose = 0)
model.summary()
loss = model.evaluate(X_test, y_test, verbose=1)
print("Mean Squared Error:", loss)
print("Root Mean Squared Error:", np.sqrt(loss))
y_pred = model.predict(X_test)
import numpy as np
import seaborn as sns
sns.residplot(y_test, y_pred, lowess=True, color="g")

from matplotlib import pyplot
pyplot.scatter(y_test, y_pred)
y_pred_1 = y_pred.flatten()
import numpy as np
import scipy.stats
corr , _ = scipy.stats.pearsonr(y_test, y_pred_1)
print("Pearsons correlation:", corr)
r2 = np.power(corr,2)
print(r2)
from scipy import stats
import numpy as np
slope, intercept, r_value, p_value, std = stats.linregress(y_test,y_pred_1)
print(slope, intercept)
