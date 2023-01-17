import pandas as pd

import numpy as np

insurance =pd.read_csv("../input/insurance/insurance.csv")

insurance.head()
insurance.shape
insurance.info()
insurance.describe()
import matplotlib.pyplot as plt

import seaborn as sns

sns.pairplot(insurance)

plt.show()
plt.figure(figsize=(20,12))

plt.subplot(1,3,1)

sns.boxplot(x="sex",y="charges",data=insurance)



plt.subplot(1,3,2)

sns.boxplot(x="smoker",y="charges",data=insurance)



plt.subplot(1,3,3)

sns.boxplot(x="region",y="charges",data=insurance)
sns.heatmap(insurance.corr(),cmap="YlGnBu", annot = True)

plt.show()
X = insurance ['age']

y = insurance ['charges']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, test_size = 0.3, random_state = 100)
X_train.head()
y_train.head()
import statsmodels.api as sm
X_train_sm = sm.add_constant(X_train)



# Fit the resgression line using 'OLS'

lr = sm.OLS(y_train, X_train_sm).fit()
lr.params
lr.summary()
plt.scatter(X_train, y_train)

plt.plot(X_train, 2707.0518 + 267.7401*X_train, 'r')

plt.show()
y_train_pred = lr.predict(X_train_sm)

res = (y_train - y_train_pred)
fig = plt.figure()

sns.distplot(res, bins = 15)

fig.suptitle('Error Terms', fontsize = 15)                  # Plot heading 

plt.xlabel('y_train - y_train_pred', fontsize = 15)         # X-label

plt.show()
plt.scatter(X_train,res)

plt.show()