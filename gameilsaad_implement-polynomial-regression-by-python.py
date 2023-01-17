import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv("../input/position-salaries/Position_Salaries.csv")
x=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,-1].values
dataset.head()
x
y
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(x,y)
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
x_poly=poly_reg.fit_transform(x)
lin_reg_2=LinearRegression()
lin_reg_2.fit(x_poly,y)
plt.scatter(x,y,color='red')
plt.plot(x,lin_reg.predict(x),color='blue')
plt.title('Truth or Bluff(Linear Regression)')
plt.xlabel('position level')
plt.ylabel('Salary')
plt.show()
plt.scatter(x,y,color='red')
plt.plot(x,lin_reg_2.predict(x_poly),color='blue')
plt.title('Truth or Bluff(Polynomial Regression)')
plt.xlabel('position level')
plt.ylabel('Salary')
plt.show()
x_grid=np.arange(min(x),max(x),0.1)
x_grid=x_grid.reshape(len(x_grid),1)
plt.scatter(x,y,color='red')
plt.plot(x_grid,lin_reg_2.predict(poly_reg.fit_transform(x_grid)),color='blue')
plt.title('Truth or Bluff(Polynomial Regression with Smoother line)')
plt.xlabel('position level')
plt.ylabel('Salary')
plt.show()
lin_reg.predict([[6.5]])
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))