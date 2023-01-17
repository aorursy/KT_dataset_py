print("Hello Jupyter notebook")
import pandas as pd



data = pd.read_csv("../input/salary.csv")



age = data.iloc[:, 0:1]

salary = data.iloc[:, 1:2]



print(age)

print()



print(salary)

print()
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(age, salary)
import matplotlib.pyplot as plt



predicted_salary = lr.predict(age);



plt.plot(age, predicted_salary)

plt.scatter(age, salary)

plt.show()