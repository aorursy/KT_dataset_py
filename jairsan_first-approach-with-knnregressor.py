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
import pandas as pd

import csv

from sklearn.neighbors import KNeighborsRegressor

from sklearn import preprocessing
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

categorical=["X0","X1","X2","X3","X4","X5","X6","X8"]
#Remove ID, store it in separate variable in case of test data

train=train.drop("ID",axis=1)

labels=test["ID"]

test=test.drop("ID",axis=1)
#Transform categorical data

for l in categorical:

    ec = preprocessing.LabelEncoder()

    ec.fit(list(train[l].values)+list(test[l].values))

    train[l]=ec.transform(list(train[l].values))

    test[l]=ec.transform(list(test[l].values))

    train[l] = train[l].astype(float)

    test[l] = test[l].astype(float)
ytrain=train["y"].as_matrix()

train=train.drop("y",axis=1)



xtrain=train.as_matrix()



#Scale data

scaler = preprocessing.MinMaxScaler()

xtrain = scaler.fit_transform(xtrain)



#Prepare the KNNRegressor

regressor=KNeighborsRegressor(10,weights='distance')

regressor.fit(xtrain,ytrain)
#Predict and save the results

predictions=regressor.predict(scaler.transform(test.as_matrix()))



with open('results2.csv', 'w', newline='') as csvfile:

    writer = csv.writer(csvfile, delimiter=',')

    writer.writerow(["ID","y"])

    for i in range(0,len(predictions)):

        writer.writerow([labels[i],predictions[i]])