from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
x = np.array([1,2,3,6])
y = np.array([2,4,6,12])
plt.scatter(x,y)
reg = LinearRegression()
# we need matrix not an array so the following
X = x.reshape(-1,1)
reg.fit(X, y)
a = reg.coef_
b = reg.intercept_
print("Regression calculated for equation y=ax+b. Params are a (slope)=%.1f, b (intercept)=%.1f" % (a,b))
print("Score is %f" % reg.score(X, y))