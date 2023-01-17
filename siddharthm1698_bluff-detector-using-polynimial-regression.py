import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('../input/Salary_Data.csv')
X = dataset.iloc[:, 0:1].values  # We use this python function to get the 1st coloumn alone as it is X
y = dataset.iloc[:, 1].values # We use this to get the 2nd coloumn alone which is the one we want to predict Y
from sklearn.linear_model import LinearRegression 
lin_reg  = LinearRegression()
lin_reg.fit(X,y)
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 5) # we use the degree as 5 here for best prediction.
X_poly = poly_reg.fit_transform(X)


lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,y)
plt.scatter(X,y,color = 'red')
plt.plot(X,lin_reg.predict(X),color = 'blue')
plt.title('truth or bluff - linear reg')
plt.xlabel('Experience level')
plt.ylabel('salary')
plt.show()



plt.scatter(X,y,color = 'red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color = 'blue')
plt.title('truth or bluff - polynomial reg')
plt.xlabel('Experience level')
plt.ylabel('salary')
plt.show()

lin_reg2.predict(poly_reg.fit_transform([[6.5]]))
