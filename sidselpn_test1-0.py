# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data_train = pd.read_csv('../input/train.csv')

data_test = pd.read_csv('../input/test.csv')

data_train.head(6)
data_train.Sex.replace(['male','female'],[1,2], inplace=True)
data_train.isnull().sum()
data_train.Age = data_train.Age.fillna(data_train.Age.median())
data_train.describe()
from sklearn.neural_network import MLPRegressor

X = pd.DataFrame(data_train.iloc[:,[1,2,4]])

X_train = X[0:800]

X_test = X[800:]



y = data_train.Age

y_train = y[0:800]

y_test = y[800:]



NN = MLPRegressor()

NN.fit(X_train,y_train)



NN_predicted = NN.predict(X_test)

R2 = NN.score(X_test,y_test)

print(R2)







891*0.9
