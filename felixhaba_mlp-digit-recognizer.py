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
data = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

X = data.drop("label", axis = 1)

y = data.label
X
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

X_train, X_val, y_train, y_val = train_test_split(X,y)

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_val_scaled = scaler.transform(X_val)

model = MLPClassifier(hidden_layer_sizes=[800,800])

model.fit(X_train_scaled, y_train)

model.score(X_val_scaled, y_val)
predictions = pd.DataFrame({"label": model.predict(X_val)}, index = X_val.index+1)

predictions

model.score(X_train, y_train)
X = scaler.fit_transform(X)

final_index = test.index

test = scaler.transform(test)

model.fit(X,y)

predictions = pd.DataFrame({"Label": model.predict(test)}, index = final_index+1)

predictions.index.rename("ImageId", inplace= True)

predictions

predictions.to_csv("submission.csv")