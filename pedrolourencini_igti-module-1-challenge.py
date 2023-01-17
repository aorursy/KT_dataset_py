#Imports

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

dataset = pd.read_csv('../input/bike-sharing-igti-challenge/comp_bikes_mod.csv')
dataset.head()
nans = dataset.isna().sum()

nans[nans > 0]
dataset.fillna(dataset.mean(), inplace = True)

dataset.head()
dataset.describe()
dataset.boxplot(["weekday", "temp"])
dataset.info()
len(dataset.dtypes.unique())
perc_null = (dataset["temp"].isna().sum() / len(dataset)) * 100

print(perc_null, "\nOr: ",round(perc_null), "%\n")
dataset.dropna(subset=["dteday"], inplace = True)

dataset.shape
dataset["temp"].mean()
dataset["windspeed"].std()
season_category = dataset["season"].astype('category')

season_category.dtype
dteday_datetime = pd.to_datetime(dataset["dteday"], format="%Y/%m/%d")

dteday_datetime.max()
dataset.boxplot(["windspeed"])
#df = pd.DataFrame(dataset, columns = ["season", "temp", "atemp", "hum", "windspeed", "cnt"])

#df.corr()

#Or

df = dataset[["season", "temp", "atemp", "hum", "windspeed", "cnt"]].corr()
plt.figure(figsize = (10, 10))

sns.heatmap(df, annot = True)

plt.title("Correlation between some columns")

plt.show()
dataset2 = dataset.copy()



dataset2.fillna(dataset[["hum", "cnt", "casual"]].mean(), inplace = True)

dataset2[["hum", "cnt", "casual"]].isna().sum()
from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score



dataset_x = dataset2[["hum", "casual"]].values

dataset_y = dataset2["cnt"].values
reg = LinearRegression()

coef = reg.fit(dataset_x, dataset_y)
previsao = reg.predict(dataset_x)
print("Y = {} X {}".format(reg.coef_, reg.intercept_))



r_2 = r2_score(dataset_y, previsao)



print("Linear Regression - R2 = ", r_2)
from sklearn.tree import DecisionTreeRegressor
clf = DecisionTreeRegressor()



clf = clf.fit(dataset_x, dataset_y)

clf = clf.predict(dataset_x)
tree_r2 = r2_score(dataset_y, clf)



print("Decision Tree Regressor - r2 : ", tree_r2)