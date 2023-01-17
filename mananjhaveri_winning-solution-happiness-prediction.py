import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv(r"/kaggle/input/now-mozilla-club-mpstme/train.csv")

test = pd.read_csv(r"/kaggle/input/now-mozilla-club-mpstme/test.csv")



y = train["Happiness Score"]

train.drop(["Happiness Score", "Country"], axis = 1, inplace = True)
train.shape, test.shape
train.head()
train.info()
train.describe()
test.info()
# Checking correlations

temp = train.copy()

temp["Happiness Score"] = y.copy()

ax = sns.heatmap(temp.corr(), annot=True)

plt.title("checking correlation")

plt.show()
ax = train.hist(bins=50, figsize=(20, 15))

plt.title("distributions")

plt.show()
from sklearn.preprocessing import StandardScaler



class prepare:



    def __init__(self):

        return



    def transform(self, df):





        df_ret = df.copy()



        # fill na with median

        df_ret["Trust"] = df_ret["Trust"].fillna(df_ret["Trust"].median())



        # scaling

        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()

        df_ret = scaler.fit_transform(df_ret)



        return df_ret
import random



class GDLinearRegressionModel():



    def __init__(self, learning_rate = 0.01, iterations = 2000):

        self.eta = learning_rate

        self.iterations = iterations





    def computeCostLinear(self, x, y, w, b, m):

        sum = 0

        for i, j in zip(x, y):

            temp = i * w + b - j

            sum += temp ** 2

        return 1/(2*m) * sum



    def computeCostMulti(self, X, y, w, b, m):

        J = 1/(2*m) * np.sum((X.dot(w) + b - y)**2)

        return J



    def get_wb_linear(self, X, y, m):

        self.history = []

        w, b = 0, 0



        for i in range(self.iterations):

            temp = b - self.eta * 1/m * np.sum(((w * X + b) - y))

            a = (X.T.dot((w * X + b) - y))

            w = w - self.eta * 1/m * (X.T.dot((w * X + b) - y))

            b = temp

            self.history.append(self.computeCostLinear(X, y, w, b, m))

        self.intercept, self.slope = b, w

        return





    def get_wb_multi(self, X, y, m):

        b = np.array([0])

        w = np.array([0] * len(X.T))

        self.history = []

        for i in range(self.iterations):

            temp = b - self.eta * 1/m * np.sum(((X.dot(w) + b) - y))

            w = w - (self.eta * 1/m * X.T.dot((X.dot(w) + b) - y))

            b = temp

            self.history.append(self.computeCostMulti(X, y, w, b, m))

        self.intercept, self.slope = b, w

        return



    def fit(self, X, y):

        m = len(y)

        try:

            lx = len(X[0])

            try:

                self.get_wb_multi(X.T, y, m)

            except:

                print("error in multi")

        except:

            self.get_wb_linear(X, y, m)

        return



    def predict(self, X, w = None, b = None):

        if w == None:

            w = self.slope

            b = self.intercept



        return X.dot(w) + b

# preprocessing

pre = prepare()

X = pre.transform(train)
# fit model

model = GDLinearRegressionModel(learning_rate = 0.001, iterations = 3000)

model.fit(X.T, y)



print("Slope is", model.slope)

print("Intercept is", model.intercept)
# plot error

from sklearn.metrics import mean_absolute_error

print("mae with predicted wights", mean_absolute_error(y, model.predict(X)))

plt.plot(model.history)

plt.title("Cost vs iterations")

plt.xlabel("iterations")

plt.ylabel("cost")

plt.show()
# Final predictions

test_countries = test["Country"]

test.drop("Country", axis = 1, inplace = True)

pre_test = pre.transform(test)

preds = model.predict(pre_test)

submission = pd.DataFrame({"Country": test_countries, "Happiness Score": preds})

submission.to_csv(r"submission.csv", index = False)