# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.cross_validation import train_test_split

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
testing = pd.read_csv("../input/test.csv")

testing['SalePrice'] = -1

training = pd.read_csv("../input/train.csv")

pd.read_csv("../input/sample_submission.csv").head()

all_sets = pd.concat([testing, training])



df = pd.get_dummies(all_sets)

df = df.fillna(df.mean())



sample = df[df.SalePrice > 0]

test = df[df.SalePrice == -1]

test = test.drop("SalePrice", axis = 1)

Xtrain, Xtest, ytrain, ytest = train_test_split(sample.drop("SalePrice", axis = 1), sample.SalePrice)

#Xtrain.head()

#Xtest.head()

#ytrain.head()

#ytest.head()
def rmse(predictions, targets):

    return np.sqrt(((predictions - targets) ** 2).mean())

from sklearn.ensemble import RandomForestRegressor



regressor = RandomForestRegressor()

regressor.fit(Xtrain, ytrain)



test.head()

predictions = regressor.predict(test)

print(predictions)

sample = pd.read_csv("../input/sample_submission.csv")
sample.SalePrice = predictions
sample.to_csv("submission.csv", index = False)
%ls