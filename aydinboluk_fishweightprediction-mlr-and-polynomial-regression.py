# Import required libraries

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use("seaborn")



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Import data



data = pd.read_csv("/kaggle/input/fish-market/Fish.csv")
# Information about the data



data.info()
# Check if there is any missing data



data.isnull().sum()
data.head()
# Summary statistics



data.describe()
data["Species"].value_counts()
sns.countplot("Species", data=data)
grp = data.groupby("Species").mean()

grp
# Barplot for mean weight of species



sns.barplot(grp.index, "Weight", data=grp)

plt.title("Mean Weight of Species")
# Box plot for weight of species



sns.boxplot("Species", "Weight", data=data)
# Calculate correlation and store it

corr = data.corr()

# Color palette for heatmap

cmap = sns.diverging_palette(20, 220, as_cmap=True)

sns.heatmap(corr, vmin=-1, vmax=1, cmap=cmap, annot=True)

plt.title("Correlation Heatmap")
# Detect outliers





def outliers(data):

    """

    Plot a boxplot and return outliers of the given data

    """

    # plot data

    sns.boxplot(data)



    # find outliers

    q1 = data.quantile(0.25)

    q3 = data.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - (iqr * 1.5)

    upper_bound = q3 + (iqr * 1.5)

    outliers = data[(data < lower_bound) | (data > upper_bound)]

    if len(outliers) != 0:

        return outliers

    return "No outliers found"



outliers(data["Weight"])
outliers(data["Length1"])
outliers(data["Length2"])
outliers(data["Length3"])
outliers(data["Width"])
# Drop outliers



data.drop([142, 143, 144], inplace=True)
# Plot linear relationship of each variable with Weight



sns.pairplot(data=data, x_vars=["Length1", "Length2", "Length3", "Height", "Width"], y_vars=[

             "Weight"], kind="reg", height=8, aspect=.5)
# Split the data into train and test data

x = data[["Length1", "Length2", "Length3", "Height", "Width"]]

y = data["Weight"]



X_train, X_test, y_train, y_test = train_test_split(

    x, y, random_state=0, test_size=0.2)

lin_reg = LinearRegression()

model = lin_reg.fit(X_train, y_train)

y_hat = lin_reg.predict(X_test)

# Calculate R-Squared and Mean Squared Error



print(f"R-Squared: {r2_score(y_test, y_hat)}")

print(f"MSE: {mean_squared_error(y_test, y_hat)}")
Input = [('polynomial', PolynomialFeatures(degree=2)),

         ('model', LinearRegression())]

pipe = Pipeline(Input)

pipe.fit(X_train, y_train)

y_hat_pipe = pipe.predict(X_test)
# Calculate R-Squared and Mean Squared Error



print(f"R-Squared: {r2_score(y_test, y_hat_pipe)}")

print(f"MSE: {mean_squared_error(y_test, y_hat_pipe)}")
# Scatterplot for linear regression

plt.figure(figsize=(10, 5))

plt.subplot("121")

plt.scatter(y_test, y_hat)

plt.xlabel("Actual Value")

plt.ylabel("Predicted Value")

plt.title("Linear Regression")

# Scatterplot for polynomial regression

plt.subplot("122")

plt.scatter(y_test, y_hat_pipe)

plt.xlabel("Actual Value")

plt.ylabel("Predicted Value")

plt.title("Polynomial Regression")



plt.suptitle("Linear vs Polynomial Regression")

plt.tight_layout()
# Create a distribution plot

# Distribution plot for linear regression

plt.figure(figsize=(10, 5))

plt.subplot("121")

sns.distplot(y_test, hist=False, label="Actual Values")

sns.distplot(y_hat, hist=False, label="Predicted Values")

plt.title("Linear Regression")

# Distribution plot for polynomial regression

plt.subplot("122")

sns.distplot(y_test, hist=False, label="Actual Values")

sns.distplot(y_hat_pipe, hist=False, label="Predicted Values")

plt.title("Polynomial Regression")



plt.suptitle("Linear vs Polynomial Regression")

plt.tight_layout()