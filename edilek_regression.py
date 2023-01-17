# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

insurance = pd.read_csv("../input/insurance-premium-prediction/insurance.csv")

f,ax = plt.subplots(figsize=(16, 16))

sns.heatmap(insurance.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
insurance.head()
plt.scatter(insurance.age,insurance.expenses)

plt.xlabel("Age")

plt.ylabel("Expenses")

plt.show()
from sklearn.linear_model import LinearRegression

linear_reg = LinearRegression()

x = pd.DataFrame(insurance.groupby('age')['expenses'].mean().index)

y = insurance.groupby('age')['expenses'].mean().values.reshape(-1,1)



linear_reg.fit(x,y)

y_=linear_reg.predict(x)

print(x)
print(y)
from sklearn.metrics import r2_score

score_DT=r2_score(y,y_)

print(score_DT)

plt.scatter(x,y,color="red")

plt.plot(x,y_,color="green")

plt.xlabel("Age")

plt.ylabel("means of Expense")

plt.title("Linear Regression of between Expense and Group of ages")

plt.show()
linear_reg.predict([[70],[75],[100]])
insurance.head()
from sklearn.ensemble import RandomForestRegressor

x=insurance.bmi.values.reshape(-1,1)

y=insurance.expenses.values.reshape(-1,1)



plt.scatter(x,y,color="red")

plt.xlabel("Body/Mass index(BMI)")

plt.ylabel("Expense")

plt.title("Correlation of between Expense and BMI")

plt.show()
random_forest=RandomForestRegressor(n_estimators=100,random_state=42)



x = pd.DataFrame(insurance.groupby('bmi')['expenses'].mean().index)

y = insurance.groupby('bmi')['expenses'].mean().values.reshape(-1,1)

random_forest.fit(x,y)

y_head=random_forest.predict(x)
from sklearn.metrics import r2_score

score_DT=r2_score(y,y_head)

print(score_DT)
plt.scatter(x,y,color="red")

plt.plot(x,y_head,color="green")

plt.xlabel("Body/Mass index(BMI)")

plt.ylabel("Expense")

plt.title("Predictions of between Expense and BMI")

plt.show()

#a sample of prediction



print(* random_forest.predict([[10],[51],[61]]))

insurance.head()
insurance_multi=pd.DataFrame()

insurance_multi["age"]=insurance.groupby('age')['expenses'].mean().index

insurance_multi["bmi"]=insurance.groupby('age')['bmi'].mean().values

insurance_multi["expenses"]=insurance.groupby('age')['expenses'].mean().values 



                
from sklearn.linear_model import LinearRegression

# df[["col1", "col3"]]

x = insurance_multi[["age", "bmi"]].values

y = insurance_multi.expenses.values.reshape(-1,1)

multiple_linear_regression = LinearRegression()

multiple_linear_regression.fit(x,y)



print("b0: ", multiple_linear_regression.intercept_)

print("b1,b2: ",multiple_linear_regression.coef_)



print(multiple_linear_regression.predict(np.array([[19,27.9],[28,33]])))


y_head=multiple_linear_regression.predict(x)
from sklearn.metrics import r2_score

score_DT=r2_score(y,y_head)

print(score_DT)


# plt.scatter(x,y,color="red")

plt.plot(x,y_head,color="green")

plt.xlabel("Body/Mass index(BMI)")

plt.ylabel("Expense")

plt.title("Predictions of between Expense and BMI")

plt.show()

y = insurance.expenses.values.reshape(-1,1)

x = insurance.bmi.values.reshape(-1,1)

from sklearn.linear_model import LinearRegression



lr = LinearRegression()



lr.fit(x,y)

y_head = lr.predict(x)



plt.plot(x,y_head,color="red",label ="linear")

plt.show()
from sklearn.preprocessing import PolynomialFeatures

polynomial_regression = PolynomialFeatures(degree = 2)



x_polynomial = polynomial_regression.fit_transform(x)





linear_regression2 = LinearRegression()

linear_regression2.fit(x_polynomial,y)



y_head2 = linear_regression2.predict(x_polynomial)



plt.plot(x,y_head2,color= "green",label = "poly")

plt.legend()

plt.show()
from sklearn.metrics import r2_score

score_DT=r2_score(y,y_head2)

print(score_DT)