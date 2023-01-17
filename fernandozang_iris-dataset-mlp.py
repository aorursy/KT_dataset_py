# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as pd

import pandas as pd



from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import mean_absolute_error, precision_score, accuracy_score



from sklearn import preprocessing as prep



print("Imported")
path = "../input/Iris.csv"

data = pd.read_csv(path)

print("Data loaded")
data.describe()

data.shape
y = data.Species

labelEncoder = prep.LabelEncoder()



y = labelEncoder.fit_transform(y).astype(np.float64)



#print(y)



columns = data.columns.drop('Species')

X = data[columns]





X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)



print("Datasets X, y defined!")
model = MLPClassifier(random_state = 1, learning_rate_init=0.003 ,max_iter=10000)

print(model)

print("Model Configured")

model.fit(X_train, y_train)

print("Model Fitted")

y_pred = model.predict(X_test)



acc_score = accuracy_score(y_test, y_pred)

print("Accuracy Score: " + str(acc_score*100) + "%")