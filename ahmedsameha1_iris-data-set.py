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
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras import utils

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score
df = pd.read_csv("../input/iris/Iris.csv")
# The id column doesn't hold any value. It's not a feature.

del df["Id"]
encoder = LabelEncoder()

# Transform the target column from string classes to integer classes

target_integer_labeled = encoder.fit_transform(df["Species"])
# One hot encode target_integer_labeled

target = utils.to_categorical(target_integer_labeled)
# Remove the target column from the training data.

del df["Species"]
def get_model():

    model = Sequential()

    model.add(Dense(8, input_dim = df.values.shape[1], activation="relu"))

    model.add(Dense(3, activation = "softmax"))

    model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])

    return model
keras_model = KerasClassifier(get_model, epochs = 200, batch_size = 5, verbose = 0)
kfold = KFold(n_splits=10, shuffle= True)
result = cross_val_score(keras_model, df, target, cv = kfold)
print("%.2f%%" % (result.mean()*100))