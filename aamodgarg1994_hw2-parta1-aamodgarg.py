from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from sklearn import linear_model as lm
o = pd.read_csv("../input/listings_detail_uploaded.csv")

o1 = o[["accommodates","price"]]

# print(o.dtypes)

# print("Correlation:\n",o1.corr())

y=o1[["price"]]

x1=o1[["accommodates"]]

# print(y)

# print(x1)

d = pd.DataFrame(np.hstack((x1,y)))

d.columns = ["x1","y"]

print(d)


model = lm.LinearRegression()

results = model.fit(x1,y)

print(model.intercept_, model.coef_)
yp2 = model.predict(x1)

print(yp2)
#Linear Regression representation using scatter plot

plt.scatter(x1,y)

plt.plot(x1,yp2, color="blue")

plt.xlabel('Number of people accommodated')

plt.ylabel('Price')

plt.title('Accommodates vs. Price')

plt.show()

#Result: Scikit-Learn

yp2 = model.predict(x1)

print(np.round(yp2[0:10],2))
print(y[0:10])

print(x1[0:10])
# Residuals = Difference of the predictions and the original values.

a1 = yp2-y

print(a1)
# calculate squares of residuals

b1 = np.square(a1)

print(b1)
# Sum the square of residuals

c1 = np.sum(b1)

print(c1)