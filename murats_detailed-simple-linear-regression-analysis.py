import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv("../input/Advertising.csv", usecols = [1,2,3,4]).copy()

df.head()
df.info()
df.describe().T
df.isnull().sum()
df.corr()
sns.pairplot(df, kind="reg");
sns.jointplot(x="TV", y="sales", data = df, kind= "reg");
# Modelling With StatsModels
import statsmodels.api as sm
X = df[["TV"]]

X[:5]
X = sm.add_constant(X)

X[:5]
y = df["sales"]

y[:5]
lm = sm.OLS(y,X)

model = lm.fit()

model.summary()
import statsmodels.formula.api as smf

lm = smf.ols("sales ~ TV", df)

model = lm.fit()

model.summary()
model.params
model.conf_int()
print("F Prob. Value : ", "%.5f" % model.f_pvalue)
print("F Statistic : ", "%.5f" % model.fvalue)
print("T Value : ", "%.5f" % model.tvalues[:1])
print("Model MSE : ", "%.3f" %  model.mse_model, "\nSales Mean : ", np.mean(df.sales))
print("Model R Squared : ", "%.5f" % model.rsquared)
print("Adjusted R Squared : ", "%.5f" % model.rsquared_adj)
model.fittedvalues[:5]
print("Sales = " + str("%.2f" % model.params[0]) + " + TV" + "*" + str("%.2f" % model.params[1]))
g = sns.regplot(df["TV"], df["sales"], ci =None, scatter_kws = {"color": "r", "s":9})

g.set_title("Model Equation: Sales = 7.03 + TV*0.05")

g.set_ylabel("Number of sales")

g.set_xlabel("TV spending")

plt.xlim(-10,310)

plt.ylim(bottom = 0);
from sklearn.linear_model import LinearRegression

X = df[["TV"]]

y = df["sales"]

reg = LinearRegression()

model = reg.fit(X,y)
model.intercept_
model.coef_
model.score(X,y)
#Prediction Phase
model.predict([[30]])
model.predict(X)[:10]
new_data = [[5],[90],[200]]

model.predict(new_data)
# Residuals 
from sklearn.metrics import mean_squared_error, r2_score
lm = smf.ols("sales ~ TV", df)

model = lm.fit()

model.summary()
mse = mean_squared_error(y, model.fittedvalues)

mse
rmse = np.sqrt(mse)

rmse
reg.predict(X)[:10]
y[:10]
comparison = pd.DataFrame({"real_y": y[:10],

                          "pred_y": reg.predict(X)[:10]})

comparison
comparison["error"] = comparison["real_y"] - comparison["pred_y"]

comparison
comparison["error_squared"] = comparison["error"]**2

comparison
np.sum(comparison["error_squared"])
np.mean(comparison["error_squared"])
np.sqrt(np.mean(comparison["error_squared"]))
model.resid[:10]
plt.plot(model.resid);