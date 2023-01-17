import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import statsmodels.api as sm

from scipy.stats import shapiro
data = pd.read_csv("../input/simple-regression/data_regression.csv")
data.head()
data.describe()
data.corr()
plt = sns.lmplot(x = "Height", y = "Weight", data = data);
stat, p = shapiro(data.Weight)

print('Statistics=%.3f, p=%.3f' % (stat, p))
sns.distplot(a = data["Weight"], hist = False);
sns.boxplot(x = data["Weight"]);
independent_variable = data[["Height"]]

independent_variable.head()
dependent_variable = data[["Weight"]]

dependent_variable.head()
linear_model = sm.OLS(dependent_variable,independent_variable)
model = linear_model.fit()
model.summary()
predicted_data = pd.DataFrame({"Weight" : data.Weight, "Predicted_Weight" : model.fittedvalues})

predicted_data
predicted_data.plot(kind='bar',figsize=(20,10))

plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')

plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

plt.show()
sns.regplot(data.Height, data.Weight);
import pandas as pd

import seaborn as sns

from sklearn.linear_model import LinearRegression
data = pd.read_csv("../input/simple-regression/data_regression.csv")
X = data[["Height"]]

y = data[["Weight"]]
reg = LinearRegression()
model = reg.fit(X,y)
model.intercept_
model.coef_
model.score(X,y)
final_df = pd.DataFrame({"Weight" : data.Weight, "Pedicted_Weight" : reg.predict(X)[:, 0]})

final_df
plt = sns.regplot(data.Height, data.Weight)

plt.set_title("Height = (-39.0449657) + (61.26229747) * Weight");
from sklearn.metrics import mean_squared_error, r2_score

mean_squared_error(y, model.predict(X))
final_df["error"] = final_df["Weight"] - final_df["Pedicted_Weight"]

final_df
stat, p = shapiro(final_df.error)

print('Statistics=%.3f, p=%.3f' % (stat, p))
sns.distplot(a = final_df["error"], hist = False);