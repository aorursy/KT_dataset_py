import numpy as np
from sklearn.linear_model import LinearRegression

x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1,1))
# reshapes to 2D, 1 col and unlimited rows.

y = np.array([5, 20, 14, 32, 22, 38])

print(x)
print(y)
# Create instance of class
model = LinearRegression()
# calculate b0 and b1, intercept and coefficients
model = model.fit(x,y)
# Getting R sqare value
# Coefficient of determination
model.score(x, y)
print('Intercept', model.intercept_) # single value
print('Slope, b1, coefficient', model.coef_) # array of coeffs
# Predict Y
y_ = model.predict(x)
print(y_)
# same as we use f(x)
y_ = model.intercept_ + model.coef_*x
print(y_)
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
# add column to statmodel
x_ = sm.add_constant(x)
print(x_) # this makes two columns, 1st for intercept and 2nd for coef
model = sm.OLS(y, x) # ordinary least square
results = model.fit() # this gives variables results
print(results.summary())
# Plot outputs
plt.scatter(x, y,  color='black')
plt.plot(x, y_, color='blue', linewidth=3)
plt.show()
df = pd.DataFrame([x.reshape(1,-1)[0],y]).transpose()
df.columns = ['x','y']
df


sns.jointplot(x='x',y='y',data=df, kind='reg')
ep = y-y_.flatten()
print('epsilons',ep)
print('mean',round(np.mean(ep)))
print('sum',round(np.sum(ep)))
print('standard deviation',round(np.std(ep),4))
sns.distplot(ep) # not a norm dist