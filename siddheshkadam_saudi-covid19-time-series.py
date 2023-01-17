# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sb

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("/kaggle/input/saudi-covid19-time-series-data/New_Saudi_Cities_COVID-19.csv")

df.head()
df.shape
df.describe()
df.dtypes
df.hist()
sb.barplot(x="Date",y="Long",data=df)
sb.barplot(x="Date",y="Lat",data=df)
sb.barplot(x="Date",y="Cases",data=df)
sb.barplot(x="Date",y="Deaths",data=df)
sb.barplot(x="Date",y="Scaled_cases",data=df)
corr=df.corr()

sb.heatmap(corr,vmax=1.,square=True)
g = sb.heatmap(df[["TotalCases","TotalRecovered","Lat","Long","Confirmed","Deaths"]].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
sb.barplot(x="TotalCases",y="TotalRecovered",data=df)
sb.barplot(x="TotalCases",y="Confirmed",data=df)
sb.barplot(x="TotalCases",y="Deaths",data=df)
plt.scatter(df["TotalCases"],df["TotalRecovered"])
covid_x=pd.DataFrame(df.iloc[:,-3])

covid_x.head()
covid_y=pd.DataFrame(df.iloc[:,-7])

covid_y.head()
from sklearn.model_selection import train_test_split



X_train, X_test, Y_train, Y_test = train_test_split(covid_x, covid_y, test_size=0.3)



from sklearn.linear_model import LinearRegression



from sklearn.tree import DecisionTreeRegressor 

regression=LinearRegression()

regression.fit(X_train, Y_train)



tree_regressor=DecisionTreeRegressor()

tree_regressor.fit(X_train, Y_train)



Y_pred_lin=regression.predict(X_test)





Y_pred_df=pd.DataFrame(Y_pred_lin, columns=["Predicted"])



Y_test.head()
Y_pred_df.head()
Y_pred_tree=tree_regressor.predict(X_test)

Y_pred_tree=pd.DataFrame(Y_pred_tree, columns=["Predicted"])

Y_pred_tree.head()

plt.figure(figsize = (5,5))

plt.title('Actual vs Predicted Salary')

plt.xlabel('TotalCases')

plt.ylabel('Confirmed')

plt.legend()

plt.scatter(list(X_test["TotalCases"]),list(Y_test["Confirmed"]),c='red')

plt.scatter(list(X_test["TotalCases"]),list(Y_pred_df["Predicted"]),c='blue')

plt.show()