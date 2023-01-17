# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt









# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train_data=pd.read_csv("../input/train.csv")

test_data=pd.read_csv("../input/test.csv")


#checking NAN

train=train_data.dropna(how="any")

test=test_data.dropna(how="any")
#plot

plt.plot(train["x"],train["y"],"ro")

plt.xlabel("input axis")

plt.ylabel("output axis")

plt.show()
x=np.array(train["x"]).reshape(-1,1)

y=np.array(train["y"]).reshape(-1,1)

x_test=np.array(test["x"]).reshape(-1,1)

y_test=np.array(test["y"]).reshape(-1,1)
from sklearn.linear_model import LinearRegression

lr=LinearRegression()

model=lr.fit(x,y)
predict=model.predict(x_test)

predict.shape
plt.plot(x_test,y_test,"*")

plt.plot(x_test,predict,"ro")

plt.xlabel("input axis")

plt.ylabel("output axis")

plt.show()
# error calculation

def MAD(x,y):

    return np.mean(np.abs(x-y))

def MSE(x,y):

    return np.mean(np.power(x-y,2))
print ("Mean of deviation:",MAD(predict,y_test))
print("MSE:",MSE(predict,y_test))