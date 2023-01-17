import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_excel("/kaggle/input/anz-synthesized-transaction/anz.xlsx")
df.head()
df.describe(include="all")
df.info()
df.count()
df.columns
df_salaries = df[df["txn_description"]=="PAY/SALARY"].groupby("customer_id").mean()

df_salaries.head()
df_salaries.nunique()
salaries = []



for customer_id in df["customer_id"]:

    salaries.append(int(df_salaries.loc[customer_id]["amount"]))

    

df["annual_salary"] = salaries
df_cus = df.groupby("customer_id").mean()

df_cus.head()
plt.figure(figsize=(15, 10))

sns.regplot("balance", "annual_salary", fit_reg=True, data=df_cus, color="g")

plt.title("Regression plot of Annual Salary against Balance", fontsize=15)

plt.xlabel("Age", fontsize=15)

plt.ylabel("Annual Salary", fontsize=15)

plt.show()
plt.figure(figsize=(15, 10))

sns.regplot("age", "annual_salary", fit_reg=True, data=df_cus, color="g")

plt.title("Regression plot of Annual Salary against Age", fontsize=15)

plt.xlabel("Age", fontsize=15)

plt.ylabel("Annual Salary", fontsize=15)

plt.show()
plt.figure(figsize=(15, 10))

sns.regplot("amount", "annual_salary", fit_reg=True, data=df_cus, color="g")

plt.title("Regression plot of Annual Salary against Amount", fontsize=15)

plt.xlabel("Age", fontsize=15)

plt.ylabel("Annual Salary", fontsize=15)

plt.show()
X = df_cus[['age', 'balance', 'amount']].values

y = df_cus['annual_salary'].values



# Splitting data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# Define model

lr = LinearRegression()



# Fit the model with training data

lr.fit(X_train, y_train)



lr.score(X, y)
# predict with test data

y_pred = lr.predict(X_test)



# Evaluating model performace

print(mean_squared_error(y_test, y_pred)) # mean squarred error

print(r2_score(y_test, y_pred)) # r2 score
# sample test case 

print("The predicted salary by the model is " + str(lr.predict([[30, 5000, 3000]])[0]))
# Define model

dt = DecisionTreeRegressor(max_depth=5,random_state=0)



# fitting model

dt.fit(X_train, y_train)



dt.score(X, y)
# Predict with test data

y_pred = dt.predict(X_test)



# Evaluating model performace

print(mean_squared_error(y_test, y_pred)) # mean squarred error

print(r2_score(y_test, y_pred)) # r2 score
# sample test case 

print("The predicted salary by the model is " + str(dt.predict([[30, 5000, 3000]])[0]))