#all necessary package imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#getting the data from csv and processing it
data = pd.read_csv("../input/age_bloodpressure.csv", index_col = "index")
data = data.iloc[:, [1,2]]
data = data[data.systolic_blood_pressure < 200]

#set variables
x = np.array(data.iloc[:, 0])
y = np.array(data.iloc[:, 1])
data.head(5)
plt.scatter(x,y)
plt.xlabel("Age")
plt.ylabel("Systolic Blood Pressure (mm Hg)")
x1, x2, y1, y2 = train_test_split(x.reshape(-1,1), y.reshape(-1,1), # reshape into column vectors
                                  random_state=0, train_size=0.50, test_size = 0.50)
model = LinearRegression(fit_intercept=True)
model.fit(x1, y1)
y2_pred = model.predict(x2)
mean_abs_error = np.sum(np.abs(y2_pred - y2))/y2.size
print("MAE: ", mean_abs_error)

R2_score = r2_score(y2, y2_pred)
print("R2: ", R2_score)

plt.scatter(x2, y2, label="test data", color = "blue")
plt.scatter(x2, y2_pred, label="predicted values", color = "red")
plt.plot(x2, y2_pred, color = "red")
plt.legend()

xfit = np.linspace(0, 80, 100)
yfit = model.predict(xfit[:, np.newaxis])
equation = "y = %sx + %s" %(model.coef_, model.intercept_)  # defining a line of best fit equation
equation = equation.replace("[", "")
equation = equation.replace("]", "")

plt.title("Age vs. Systolic Blood Pressure")
plt.xlabel("Age")
plt.ylabel("Systolic Blood Pressure (mm Hg)")
plt.scatter(x1, y1, label="Training Data")
plt.plot(xfit, yfit, label=equation)
plt.scatter(x2, y2, color = 'r', label="Testing Data")
plt.legend()

R2_score = r2_score(np.array(y).reshape(-1,1), model.predict(np.array(x).reshape(-1, 1)))
print("R2_score: ", R2_score)
#still working on the code