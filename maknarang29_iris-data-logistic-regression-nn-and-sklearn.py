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
data = pd.read_csv("/kaggle/input/iris/Iris.csv")
data.head()
data = data.drop(labels ="Id",axis =1)
targets = data["Species"].values
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

target_labels = le.fit_transform(targets)
target_labels
data = data.drop(labels = "Species",axis =1)
data = data.values
data.shape
from sklearn.linear_model import LogisticRegression

softmax_reg = LogisticRegression(solver='lbfgs',multi_class="multinomial",C = 10,max_iter = 1500)
shuffled_indices = np.random.randint(1,150,target_labels.shape[0])
data = data[shuffled_indices]

target = target_labels[shuffled_indices]
softmax_reg.fit(data[:125],target[:125])
softmax_reg.predict(data[133].reshape(1,-1))
target[133]
import tensorflow

from tensorflow import keras

from tensorflow.keras import layers
model = keras.Sequential([

    keras.layers.Flatten(input_shape = (data.shape[1],)),

    keras.layers.Dense(3,activation = "softmax")

])



model.compile(loss = "sparse_categorical_crossentropy" , optimizer  ="adam",metrics = ["accuracy"])
model.fit(data[:125],target[:125],epochs = 15,validation_split=0.2)
np.argmax(model.predict(data[137].reshape(1,-1)))
preds = []

for i in range(125,150):

    preds.append(np.argmax(model.predict(data[i].reshape(1,-1))))
preds
target[125:]
def func(x):

    if (x == 2):

        return 1

    elif (x == 1):

        return 2

    elif (x == 0):

        return 0