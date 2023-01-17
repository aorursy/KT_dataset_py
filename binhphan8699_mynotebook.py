import pandas as pd
data = pd.read_csv("/kaggle/input/salary.csv")
print(data)

ages = data.iloc[:,0:1]
salaries = data.iloc[:,1:2]

print(ages)
print(salaries)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(ages,salaries)
predicted_salary = lr.predict(ages)
print(predicted_salary)
import matplotlib.pyplot as plt 

plt.plot(ages, predicted_salary)
plt.plot(ages, salaries)
plt.scatter(ages,salaries)
plt.scatter(ages,predicted_salary)