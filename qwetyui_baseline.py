import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
ROOT = "../input/"
TRAINFILE_PATH = ROOT + "train.csv"
TESTFILE_PATH = ROOT + "test.csv"
VALIDATION_PART = 0.1
train = pd.read_csv(TRAINFILE_PATH)
test = pd.read_csv(TESTFILE_PATH)
train.head(30)
def vis(dataframe):
    for col in dataframe.drop(["Price"], axis=1):
        fig, ax = plt.subplots()
        ax.scatter(x = dataframe[col], y = dataframe['Price'])
        plt.ylabel('Price', fontsize=13)
        plt.xlabel(col, fontsize=13)
        plt.show()
vis(train)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
X = np.array(train.drop(["Price"], axis=1))
y = np.array(train["Price"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=VALIDATION_PART)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_true = y_test
rmse = ((((y_pred) - (y_true))**2).mean()) ** 0.5
print("rmsle =", rmse)
def data_prepare(data):
    data["RoomCount"] = data["RoomCount"].apply(lambda x: x / 2)
    #<...> add smth more

data_prepare(train)
data_prepare(test)
new_model = LinearRegression()
X = np.array(train.drop(["Price"], axis=1))
y = np.array(train["Price"])
new_model.fit(X, y)
commit = new_model.predict(np.array(test))
f = open("res.csv", "wt")
f.write("Id,Price\n")
id = 0
for price in commit:
    f.write(str(id + 2018) + "," + str(price) + "\n")
    id += 1
f.close()