import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm
data = pd.read_csv(
    "https://raw.githubusercontent.com/saidRaiss/dataset/master/advertising_ads"
  )
data.info()
data.head() # Display first five values (default)
# Or use the sample(n) method to display n random values.
data.drop(['Unnamed: 0'], axis=1, inplace=True)
# axis=1 it indicates that it is a column (axis=0 for index).
# inplace=True it indicates that the modification made to the data variable.
data.sample(5)
correlation = data.corr()
# correlation plot
sns.heatmap(correlation, annot= True);
# We use matplotlib, a popular Python plotting library, to create points.
plt.figure(figsize=(12, 6))
plt.scatter(
    data['TV'],
    data['sales'],
    c='red'
)
plt.xlabel("Money spent on TV ads ($)")
plt.ylabel("Sales (k$)")
plt.show()
# Input
X = data['TV'].values.reshape(-1, 1)
# Output
Y = data['sales'].values.reshape(-1, 1)
# Create a linear regression model
reg = LinearRegression()
# "fit" a data to train
reg.fit(X, Y)
print(
    "The linear model is: Y={:.5} + {:.5}X"
    .format(reg.intercept_[0], reg.coef_[0][0])
    )
predictions = reg.predict(X)
plt.figure(figsize=(12, 6))
plt.scatter(
    data['TV'],
    data['sales'],
    c='red'
)
plt.plot(
    data['TV'],
    predictions,
    c='blue',
    linewidth = 2
)
plt.xlabel("Money spent on TV ads ($)")
plt.ylabel("Sales (k$)")
plt.show()
Xm = data.drop(['sales'], axis=1)
Y = data['sales'].values.reshape(-1, 1)
reg = LinearRegression()
reg.fit(Xm, Y)
print(
    "The linear model is: Y={:.5} + {:.5}*TV + {:.5}*radio + {:.5}*newspaper"
    .format(
        reg.intercept_[0], reg.coef_[0][0], reg.coef_[0][1], reg.coef_[0][2]
        )
    )
X = np.column_stack((data['TV'], data['radio'], data['newspaper']))
Y = data['sales']
X2 = sm.add_constant(X)
est = sm.OLS(Y, X2)
est2 = est.fit()
print(est2.summary())