# basic settings
import numpy as np
import pandas as pd 
titanic_data = pd.read_csv("../input/titanic.csv")

titanic_data.head()
titanic_data.describe()
print(titanic_data.keys())
print(titanic_data.dtypes)
num_values = titanic_data.dtypes[titanic_data.dtypes == "float"]
print(num_values)
pd.isnull(titanic_data).sum()
# find any missing values and preprocess them appropriately
titanic_data = titanic_data.drop(["Cabin", "Ticket"], axis = 1)
titanic_data.head()
pd.isnull(titanic_data).sum()
print(titanic_data.groupby("Pclass").mean())
# check out any rough trends between data
pd.scatter_matrix(titanic_data[["Age", "Fare", "Pclass"]], figsize = (10,10))
titanic_data.groupby("Pclass").median()
# missing values in Age: group by class and substitute with median values
# missing values in Fare: groupby class and substitute with median value

t_agenan = titanic_data.loc[np.isnan(titanic_data["Age"])]
t_agenan_c1 = t_agenan[t_agenan["Pclass"] == 1].index
t_agenan_c2 = t_agenan[t_agenan["Pclass"] == 2].index
t_agenan_c3 = t_agenan[t_agenan["Pclass"] == 3].index
titanic_data.set_value(t_agenan_c1, "Age", 42)
titanic_data.set_value(t_agenan_c2, "Age", 26.5)
titanic_data.set_value(t_agenan_c3, "Age", 24)
pd.isnull(titanic_data).sum() # there is only one missing value left now
titanic_data.loc[np.isnan(titanic_data["Fare"])]
titanic_data.loc[152, "Fare"] = 7.8958
titanic_data.loc[152,]
pd.isnull(titanic_data).sum() # No missing values left
pd.scatter_matrix(titanic_data[["Age", "Fare", "Pclass"]], figsize = (10,10))
import matplotlib.pyplot as plt

plt.scatter(titanic_data["Pclass"], titanic_data["Fare"])
plt.show()
[a, b] = np.polyfit(titanic_data["Pclass"], titanic_data["Fare"], deg = 1)
[a, b] # slope and intercept for least square regression line
fig, ax = plt.subplots()
ax.scatter(titanic_data["Pclass"], titanic_data["Fare"])
fitline1 = b + (a * titanic_data["Pclass"])
ax.plot(titanic_data["Pclass"], fitline1, "r--")
plt.show() # for validation, we need to test F lack-of-fit test
plt.scatter(titanic_data["Pclass"], titanic_data["Age"])
plt.show()
[a1, b1] = np.polyfit(titanic_data["Pclass"], titanic_data["Age"], deg = 1)
[a1, b1] # slope and intercept for least square regression line
fig, ax = plt.subplots()
ax.scatter(titanic_data["Pclass"], titanic_data["Age"])
fitline2 = b1 + (a1 * titanic_data["Pclass"])
ax.plot(titanic_data["Pclass"], fitline2, "r--")
plt.show() # for validation, we need to test F lack-of-fit test
plt.scatter(titanic_data["Age"], titanic_data["Fare"])
plt.show()
[a2, b2] = np.polyfit(titanic_data["Age"], titanic_data["Fare"], deg = 1)
[a2, b2] # slope and intercept for least square regression line
fig, ax = plt.subplots()
ax.scatter(titanic_data["Age"], titanic_data["Fare"])
fitline3 = b2 + (a2 * titanic_data["Age"])
ax.plot(titanic_data["Age"], fitline3, "r--")
plt.show() # for validation, we need to test F lack-of-fit test