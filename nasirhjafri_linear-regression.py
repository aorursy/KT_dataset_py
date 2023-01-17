import numpy as np

import pandas as pd



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



train_dataset = pd.read_csv('../input/train.csv')

test_dataset = pd.read_csv('../input/test.csv')



train_dataset = train_dataset.dropna()

test_dataset = test_dataset.dropna()



train_x = train_dataset.iloc[:, :-1].values

train_y = train_dataset.iloc[:, 1].values



test_x = test_dataset.iloc[:, :-1].values

test_y = test_dataset.iloc[:, 1].values



from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(train_x, train_y)
predict_y = regressor.predict(test_x)



import matplotlib.pyplot as plt





plt.scatter(train_x, train_y, color="green")

plt.plot(train_x, regressor.predict(train_x), color="red")

plt.title("Training data set")

plt.show()
plt.scatter(test_x, test_y, color="green")

plt.plot(test_x, predict_y, color="red")

plt.title("Test data set")

plt.xlabel("x")

plt.ylabel("y")

plt.show()