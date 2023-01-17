import pandas as pd

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

%matplotlib inline

salary = pd.read_csv("../input/salary/Salary.csv")

salary.head(5)
salary.columns
exp = salary["YearsExperience"].values

sal = salary["Salary"].values

exp = exp.reshape(-1,1)

sal = sal.reshape(-1,1)
plt.scatter(exp,sal)

plt.xlabel("x-axis")

plt.title("Salary data")

plt.ylabel("y-axis")
x_train, x_test, y_train, y_test = train_test_split(exp,sal,train_size=0.8,test_size=0.2,random_state=100)



lm = LinearRegression()

lm.fit(x_train,y_train)

y_predicted = lm.predict(x_test)

print(f"Train Accuracy {round(lm.score(x_train,y_train)* 100,2)}%")
plt.scatter(x_train, y_train)

plt.plot(x_test,y_predicted, color = "red")