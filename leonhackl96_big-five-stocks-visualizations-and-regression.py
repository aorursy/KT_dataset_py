import numpy as np

import pandas as pd

import datetime



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib import style

from matplotlib.pyplot import plot



style.use("seaborn-whitegrid")

%matplotlib inline

%config InlineBackend.figure_format = "retina"
df = pd.read_csv("/kaggle/input/big-five-stocks/big_five_stocks.csv")
data_size_mb = df.memory_usage().sum() / 1024 / 1024

print("Data memory size: %.2f MB" % data_size_mb)
df.head(3)
df.shape
df.info()
df.name.unique()
df.isnull().sum().sort_values(ascending = False)
df["Unnamed: 0"] = pd.to_datetime(df["Unnamed: 0"])

df["year"] = df["Unnamed: 0"].dt.year

df["month"] = df["Unnamed: 0"].dt.month

df = df.rename({"Unnamed: 0" : "date"}, axis = 1, inplace = False)
sub_aapl = df[df.name == "AAPL"].drop("name", axis = 1)

sub_msft = df[df.name == "MSFT"].drop("name", axis = 1)

sub_amzn = df[df.name == "AMZN"].drop("name", axis = 1)

sub_googl = df[df.name == "GOOGL"].drop("name", axis = 1)

sub_fb = df[df.name == "FB"].drop("name", axis = 1)

sub_nas = df[df.name == "^IXIC"].drop("name", axis = 1)



sub_aapl.set_index("date", drop = True, inplace = True)

sub_msft.set_index("date", drop = True, inplace = True)

sub_amzn.set_index("date", drop = True, inplace = True)

sub_googl.set_index("date", drop = True, inplace = True)

sub_fb.set_index("date", drop = True, inplace = True)

sub_nas.set_index("date", drop = True, inplace = True)
plt.figure(figsize = (14,8))

sub_aapl["close"].plot(color = "red", label = "close")

sub_aapl["open"].plot(color = "blue", alpha = 0.7, label = "open")

plt.title("Apple", fontsize = 16)

plt.legend()

plt.show()
plt.figure(figsize = (14,8))

sub_msft["close"].plot(color = "red", label = "close")

sub_msft["open"].plot(color = "blue", alpha = 0.7, label = "open")

plt.title("Microsoft", fontsize = 16)

plt.legend()

plt.show()
plt.figure(figsize = (14,8))

sub_amzn["close"].plot(color = "red", label = "close")

sub_amzn["open"].plot(color = "blue", alpha = 0.7, label = "open")

plt.title("Amazon", fontsize = 16)

plt.legend()

plt.show()
plt.figure(figsize = (14,8))

sub_googl["close"].plot(color = "red", label = "close")

sub_googl["open"].plot(color = "blue", alpha = 0.7, label = "open")

plt.title("Google", fontsize = 16)

plt.legend()

plt.show()
plt.figure(figsize = (14,8))

sub_fb["close"].plot(color = "red", label = "close")

sub_fb["open"].plot(color = "blue", alpha = 0.7, label = "open")

plt.title("Facebook", fontsize = 16)

plt.legend()

plt.show()
plt.figure(figsize = (14,8))

sub_nas["close"].plot(color = "red", label = "close")

sub_nas["open"].plot(color = "blue", alpha = 0.7, label = "open")

plt.title("Nasdaq", fontsize = 16)

plt.legend()

plt.show()
sub_aapl_year = df[(df.name == "AAPL") & (df.year >= 2015)].reset_index().drop(["name", "index"], axis = 1)

sub_msft_year = df[(df.name == "MSFT") & (df.year >= 2015)].reset_index().drop(["name", "index"], axis = 1)

sub_amzn_year = df[(df.name == "AMZN") & (df.year >= 2015)].reset_index().drop(["name", "index"], axis = 1)

sub_googl_year = df[(df.name == "GOOGL") & (df.year >= 2015)].reset_index().drop(["name", "index"], axis = 1)

sub_fb_year = df[(df.name == "FB") & (df.year >= 2015)].reset_index().drop(["name", "index"], axis = 1)

sub_nas_year = df[(df.name == "^IXIC") & (df.year >= 2015)].reset_index().drop(["name", "index"], axis = 1)
sub_aapl_year.head(3)
comp_stocks = pd.DataFrame({"AAPL" : sub_aapl_year["close"],

                            "GOOGL" : sub_googl_year["close"],

                            "AMZN" : sub_amzn_year["close"],

                            "MSFT" : sub_msft_year["close"],

                            "FB" : sub_fb_year["close"],

                            "NASDAQ" : sub_nas_year["close"]})
comp_stocks.head(3)
sub_returns = comp_stocks.apply(lambda x: x / x[0])

sub_returns["DATE"] = sub_aapl_year["date"]

sub_returns.set_index("DATE", drop = True, inplace = True)
sub_returns.head(3)
sub_returns.plot(figsize = (14,8))

plt.show()
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures
X = sub_amzn_year[["date"]].values

Y = sub_amzn_year[["close"]].values
s = StandardScaler()

X = s.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, 

                                                    Y, 

                                                    random_state = 0, 

                                                    test_size = 0.25)
lm = LinearRegression()

lm.fit(X_train, Y_train)

print("Test score: " + str(lm.score(X_test, Y_test)))

print("Train score: " + str(lm.score(X_train, Y_train)))



Y_predicted_lm = lm.predict(X_test)
plt.figure(figsize = (14,8))

plt.scatter(X_train, Y_train, s = 10, label = "Train")

plt.scatter(X_test, Y_test, s = 10, label = "Test")

plt.plot(X_test, Y_predicted_lm, color = "green", linewidth = 2, label = "Regression")

plt.xlabel("Time")

plt.title("Linear Regression", fontsize = 16)

plt.legend()

plt.show()
poly = PolynomialFeatures(degree = 3)

X_train_transformed = poly.fit_transform(X_train)

X_test_transformed = poly.fit_transform(X_test)
lm_poly = LinearRegression()

lm_poly.fit(X_train_transformed, Y_train)

print("Test score: " + str(lm_poly.score(X_test_transformed, Y_test)))

print("Train score: " + str(lm_poly.score(X_train_transformed, Y_train)))
plt.figure(figsize = (14,8))

plt.scatter(X_train, Y_train, s = 10, label = "Train")

plt.scatter(X_test, Y_test, s = 10, label = "Test")

plt.plot(X, lm_poly.predict(poly.fit_transform(X)), color = "green", linewidth = 2, label = "Regression")

plt.xlabel("Time")

plt.title("Polynomial Regression", fontsize = 16)

plt.legend()

plt.show()