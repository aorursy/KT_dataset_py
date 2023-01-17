# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

pd.set_option('display.max_columns', 500)

df = pd.read_csv("../input/2015.csv")

df.head()
df.tail()
df.isnull().any()
df.info()
df.columns
sns.lmplot(x = "Happiness Score",y = "Economy (GDP per Capita)",data = df)
sns.lmplot(x = "Happiness Score",y = "Family",data = df)
sns.lmplot(x = "Happiness Score",y = 'Health (Life Expectancy)',data = df)
sns.lmplot(x = "Happiness Score",y = 'Freedom',data = df)
sns.lmplot(x = "Happiness Score",y = 'Trust (Government Corruption)',data = df)
sns.lmplot(x = "Happiness Score",y = "Generosity",data = df)
sns.lmplot(x = "Happiness Score",y = "Dystopia Residual",data = df)
x_df = df.drop(columns = ["Happiness Rank","Standard Error"],axis = 1)

x_df.corr()
plt.figure(figsize = (12,8))

sns.heatmap(x_df.corr(),cmap =  "inferno",annot = True)

plt.show()
group = x_df.groupby(by = "Region")

new_df = group.mean()

new_df
new1_df = group.median()

new1_df
plt.figure(figsize = (30,12))

sns.barplot(x = x_df["Region"],y = x_df["Happiness Score"],data = df)

plt.xlabel("Region",fontsize = 20)

plt.ylabel("Happiness Score",fontsize = 20)
plt.figure(figsize = (30,12))

sns.barplot(x = x_df["Region"],y = x_df["Economy (GDP per Capita)"],data = df)

plt.xlabel("Region",fontsize = 20)

plt.ylabel("Economy (GDP per Capita)",fontsize = 20)
plt.figure(figsize = (30,12))

sns.barplot(x = x_df["Region"],y = x_df["Family"],data = df)

plt.xlabel("Region",fontsize = 20)

plt.ylabel("Family",fontsize = 20)
plt.figure(figsize = (30,12))

sns.barplot(x = x_df["Region"],y = x_df["Health (Life Expectancy)"],data = df)

plt.xlabel("Region",fontsize = 20)

plt.ylabel("Health (Life Expectancy)",fontsize = 20)
plt.figure(figsize = (30,12))

sns.barplot(x = x_df["Region"],y = x_df["Freedom"],data = df)

plt.xlabel("Region",fontsize = 20)

plt.ylabel("Freedom",fontsize = 20)
plt.figure(figsize = (30,12))

sns.barplot(x = x_df["Region"],y = x_df["Trust (Government Corruption)"],data = df)

plt.xlabel("Region",fontsize = 20)

plt.ylabel("Trust (Government Corruption)",fontsize = 20)
plt.figure(figsize = (30,12))

sns.barplot(x = x_df["Region"],y = x_df["Generosity"],data = df)

plt.xlabel("Region",fontsize = 20)

plt.ylabel("Generosity",fontsize = 20)
plt.figure(figsize = (30,12))

sns.barplot(x = x_df["Region"],y = x_df["Dystopia Residual"],data = df)

plt.xlabel("Region",fontsize = 20)

plt.ylabel("Dystopia Residual",fontsize = 20)
plt.figure(figsize = (12,8))

sns.scatterplot(x = "Economy (GDP per Capita)",y = "Happiness Score",data = df,hue = "Region")
plt.figure(figsize = (12,8))

sns.scatterplot(x = "Family",y = "Happiness Score",data = df,hue = "Region")
plt.figure(figsize = (12,8))

sns.scatterplot(x = "Health (Life Expectancy)",y = "Happiness Score",data = df,hue = "Region")
plt.figure(figsize = (12,8))

sns.scatterplot(x = "Freedom",y = "Happiness Score",data = df,hue = "Region")
plt.figure(figsize = (12,8))

sns.scatterplot(x = "Trust (Government Corruption)",y = "Happiness Score",data = df,hue = "Region")
plt.figure(figsize = (12,8))

sns.scatterplot(x = "Generosity",y = "Happiness Score",data = df,hue = "Region")
plt.figure(figsize = (12,8))

sns.scatterplot(x = "Dystopia Residual",y = "Happiness Score",data = df,hue = "Region")
X = df.drop(["Country","Region","Happiness Rank","Happiness Score"],axis = 1).values

y = df["Happiness Score"].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)

print("accuracy is: "+ str(lr.score(X_test,y_test) * 100) + "%")

print("Mean Absolute Error: {}".format(mean_absolute_error(y_test,y_pred)))

print("Mean Squared Error: {}".format(mean_squared_error(y_test,y_pred)))

print("R Squared: {}".format(r2_score(y_test,y_pred)))
from sklearn import linear_model

las = linear_model.Lasso()

las.fit(X_train,y_train)

y_pred = las.predict(X_test)

print("accuracy is: "+ str(las.score(X_test,y_test) * 100) + "%")

print("Mean Absolute Error: {}".format(mean_absolute_error(y_test,y_pred)))

print("Mean Squared Error: {}".format(mean_squared_error(y_test,y_pred)))

print("R Squared: {}".format(r2_score(y_test,y_pred)))
rig = linear_model.Ridge()

rig.fit(X_train,y_train)

y_pred_1 = rig.predict(X_test)

print("Accuracy is: "+ str(rig.score(X_test,y_test) * 100) + "%")

print("Mean Absolute Error: {}".format(mean_absolute_error(y_test,y_pred_1)))

print("Mean Squared Error: {}".format(mean_squared_error(y_test,y_pred_1)))

print("R Squared: {}".format(r2_score(y_test,y_pred_1)))
from sklearn.tree import DecisionTreeRegressor

tree_ = DecisionTreeRegressor()

tree_.fit(X_train,y_train)

y_pred_2 = tree_.predict(X_test)

print("Accuracy is: "+ str(tree_.score(X_test,y_test) * 100) + "%")

print("Mean Absolute Error: {}".format(mean_absolute_error(y_test,y_pred_2)))

print("Mean Squared Error: {}".format(mean_squared_error(y_test,y_pred_2)))

print("R Squared: {}".format(r2_score(y_test,y_pred_2)))
from sklearn.ensemble import AdaBoostRegressor

ada = AdaBoostRegressor(loss = "exponential")

ada.fit(X_train,y_train)

y_pred_3 = ada.predict(X_test)

print("Accuracy is: "+ str(ada.score(X_test,y_test) * 100) + "%")

print("Mean Absolute Error: {}".format(mean_absolute_error(y_test,y_pred_3)))

print("Mean Squared Error: {}".format(mean_squared_error(y_test,y_pred_3)))

print("R Squared: {}".format(r2_score(y_test,y_pred_3)))