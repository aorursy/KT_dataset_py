import numpy as np

import pandas as pd

from sklearn import datasets, linear_model

import os

os.listdir("../input")
train = pd.read_csv("../input/train.csv", index_col=0)

test = pd.read_csv("../input/test.csv", index_col=0)

train.head(5)
def func(v):

    if (v == "male"):

        return 0

    return 1



x = train["Sex"].map(func).values.reshape(-1, 1)

y = train["Survived"].values.reshape(-1, 1)

lr = linear_model.LinearRegression().fit(x, y)



def func2(v):

    if (v == 1):

        return 1

    if (v == 2):

        return 0.5

    return 0



x1 = train["Pclass"].map(func2).values.reshape(-1, 1)

y1 = train["Survived"].values.reshape(-1, 1)

lr1 = linear_model.LinearRegression().fit(x1, y1)



def func3(v):

    if (v >= 0.0 and v <= 14.0):

        return 0.3

    if (v > 14.0 and v <= 18.0 ):

        return 0.4

    if (v > 18.0 and v <= 30.0):

        return 0.6

    if (v > 30.0 and v <= 40.0):

        return 0.5

    if (v > 40.0 and v <= 60.0):

        return 0.3

    if (v > 60.0):

        return 0.1

    return 0.2



x2 = train["Age"].map(func3).values.reshape(-1, 1)

y2 = train["Survived"].values.reshape(-1, 1)

lr2 = linear_model.LinearRegression().fit(x2, y2)



test_x = test["Sex"].map(func).values.reshape(-1, 1)

test_x1 = test["Pclass"].map(func2).values.reshape(-1, 1)

test_x2 = test["Age"].map(func3).values.reshape(-1, 1)



result = lr.predict(test_x)

result2 = lr1.predict(test_x1)

result3 = lr2.predict(test_x2)



#result = lr.predict(x)

#result2 = lr1.predict(x1)

#result3 = lr2.predict(x2)



for i in range(0, result.size):

    result[i] += result2[i]

    result[i] += result3[i]

    result[i] /= 3



def func1(v):

    if (v <= 0.5):

        return 0

    return 1



res = result.flatten().tolist()

res = list(map(func1, res))

train_test = train["Survived"].values



sum1 = 0

for i in range(0, train_test.size):

    if (res[i] == train_test[i]):

        sum1 += 1



#sum1 / train_test.size



df = pd.DataFrame({'PassengerId':pd.read_csv('../input/test.csv')['PassengerId'].values, 'Survived':res}, dtype=int)

df.to_csv('submission.csv', index=False)