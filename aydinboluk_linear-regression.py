import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

plt.style.use("seaborn")
data = pd.read_csv("../input/linear-regression-dataset/Linear Regression - Sheet1.csv")
data.info()
data.head()
data.isnull().sum()
cmap = sns.diverging_palette(20, 220, as_cmap=True)

sns.heatmap(data.corr(), cmap=cmap, vmin=-1, vmax=1, annot=True)
sns.regplot(x="X", y="Y", data=data)
lm = LinearRegression()

x = data[["X"]]

y = data[["Y"]]

lm.fit(x, y)

predict = lm.predict(x)
print(f"Intercept of the model: {lm.intercept_}")

print(f"Slope of the model: {lm.coef_}")
ax = sns.distplot(y, hist=False, label="Observed Values")

sns.distplot(predict, hist=False, label="Fitted Values", ax=ax)

plt.title("Observed vs Fitted Values")

plt.xlabel("X")

plt.ylabel("Y")
sns.residplot(data["X"], data["Y"])
print(f"The R-Squared is: {lm.score(x, y)}")

print(f"The MSE is: {mean_squared_error(predict, y)}")