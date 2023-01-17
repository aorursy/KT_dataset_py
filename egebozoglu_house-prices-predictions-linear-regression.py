import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # data visualization

import seaborn as sns

sns.set()

import statsmodels.api as sm

from sklearn.linear_model import LinearRegression
dirty = pd.read_csv("../input/housePractice.csv")

dirty.head()
dirty.describe()
dirty.columns.values
data = dirty.drop(['id', 'date','bedrooms', 'bathrooms', 'view', 'grade',

       'sqft_above', 'sqft_basement', 'yr_renovated',

       'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15','sqft_lot','waterfront'], axis=1)

data.head()
data.rename(columns={"sqft_living":"size", "floors":"floor","yr_built":"year"}, inplace=True)

data.head()
data.isnull().sum()
sns.distplot(data["price"])
q = data["price"].quantile(0.99)

data_1 = data[data["price"]<q]
sns.distplot(data_1["price"])
sns.distplot(data_1["year"])
sns.distplot(data_1["floor"])
sns.distplot(data_1["condition"])
data_cleaned = data_1.reset_index(drop=True)

data_cleaned
f, (ax1,ax2,ax3,ax4) = plt.subplots(1,4, sharey=True, figsize=(16,4))

ax1.scatter(data_cleaned["size"],data_cleaned["price"])

ax1.set_title("Size and Price")

ax2.scatter(data_cleaned["floor"], data_cleaned["price"])

ax2.set_title("Floor and Price")

ax3.scatter(data_cleaned["condition"], data_cleaned["price"])

ax3.set_title("Condition and Price")

ax4.scatter(data_cleaned["year"], data_cleaned["price"])

ax4.set_title("Year and Price")



plt.show()
log_price = np.log(data_cleaned["price"])

data_cleaned["log_price"] = log_price

data_cleaned
f, (ax1,ax2,ax3,ax4) = plt.subplots(1,4, sharey=True, figsize=(16,4))

ax1.scatter(data_cleaned["size"],data_cleaned["log_price"])

ax1.set_title("Size and Log Price")

ax2.scatter(data_cleaned["floor"], data_cleaned["log_price"])

ax2.set_title("Floor and Log Price")

ax3.scatter(data_cleaned["condition"], data_cleaned["log_price"])

ax3.set_title("Condition and Log Price")

ax4.scatter(data_cleaned["year"], data_cleaned["log_price"])

ax4.set_title("Year and Log Price")



plt.show()
data_cleaned.drop("price",axis=1, inplace=True)

data_cleaned.head()
data_cleaned.columns.values
from statsmodels.stats.outliers_influence import variance_inflation_factor

variables = data_cleaned[['size', 'floor', 'condition', 'year']]

vif = pd.DataFrame()

vif["VIF"] = [variance_inflation_factor(variables.values,i) for i in range(variables.shape[1])]

vif["Features"] = variables.columns
vif
data_no_mc = data_cleaned.drop("year",axis=1)
data_no_mc.columns.values
cols = ['log_price','size', 'floor', 'condition']
data_preprocessed = data_no_mc[cols]

data_preprocessed
target = data_preprocessed["log_price"]

inputs = data_preprocessed.drop("log_price",axis=1)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(inputs)
inputs_scaled = scaler.transform(inputs)
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(inputs_scaled, target, test_size=0.2, random_state=31)
reg = LinearRegression()

reg.fit(x_train,y_train)
y_hat = reg.predict(x_train)
plt.scatter(y_train, y_hat)

plt.xlabel("Targets (y_train)", size=18)

plt.ylabel("Predictions (y_hat)", size=18)

plt.xlim(11,15.5)

plt.ylim(11,15.5)

plt.show()
sns.distplot(y_train - y_hat)

plt.title("Residuals PDF",size=18)
reg.score(x_train, y_train)
reg.intercept_
reg.coef_
reg_summary = pd.DataFrame(inputs.columns.values,columns=["Features"])

reg_summary["Weights"] = reg.coef_

reg_summary
y_hat_test = reg.predict(x_test)
plt.scatter(y_test, y_hat_test, alpha=0.2)

plt.xlabel("Targets (y_test)",size=18)

plt.ylabel("Predictions (y_hat_test)",size=18)

plt.xlim(11,15.5)

plt.ylim(11,15.5)

plt.show()
df_pf = pd.DataFrame(np.exp(y_hat_test), columns=["Predictions"])

df_pf.head()
y_test = y_test.reset_index(drop=True)
df_pf["Prices"] = np.exp(y_test)

df_pf.head()
df_pf["Residual"] = df_pf["Prices"] - df_pf["Predictions"]

df_pf["Difference%"] = np.absolute(df_pf["Residual"]/df_pf["Prices"]*100)

df_pf
pd.options.display.max_rows = 4500

pd.set_option("display.float_format", lambda x: "%.2f" % x)

df_pf.sort_values(by=["Difference%"])