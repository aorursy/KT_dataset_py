# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/column_2C_weka.csv')
data.head()
data.info()
data['class'].value_counts()
d_stat = data.describe()

d_stat
data.corr()
x_names = list(d_stat.columns) # Using the columns for the each orthopedic status'names.

y_values = list(d_stat.mean()) # Using the mean values for the each feature.

plt.figure(figsize=(10,5))

sns.barplot(x=x_names, y=y_values)

plt.xticks(rotation=45)

plt.ylabel('Mean values of 310 the patients')

plt.show()
# Filtering only abnormal patients:

d_new = data[data['class'] == 'Abnormal']



# Our graphics x will be pelvic incidence and y will be sacral slope values:

x = np.array(d_new.loc[:,'pelvic_incidence']).reshape(-1,1)

y = np.array(d_new.loc[:, 'sacral_slope']).reshape(-1,1)
# Now visualize the values:

plt.figure(figsize=(9,9))

plt.scatter(x=x, y=y, color='purple', alpha=0.3)

plt.xlabel('Pelvic Incidence Values')

plt.ylabel('Sacral Slope Values')

plt.show()
from sklearn.linear_model import LinearRegression



lr = LinearRegression()

lr.fit(x,y)
# Predicting new y values from our data's x values:

y_head = lr.predict(x)



plt.figure(figsize=(9,9))

plt.scatter(x,y, color='purple', alpha=0.3, label='patients')

plt.plot(x, y_head, color='blue', label='linear')

plt.xlabel('Pelvic Incidence Values')

plt.ylabel('Sacral Slope Values')

plt.legend()

plt.show()
print('Sacral Slope Value of the Patient with 100 Pelvic Incidence Value: ', lr.predict([[110]]))
from sklearn.metrics import r2_score

print('Our models r square score is: ', r2_score(y, y_head))
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)



x_poly = poly.fit_transform(x)



lr2 = LinearRegression()

lr2.fit(x_poly, y)

y_head2 = lr2.predict(x_poly)
plt.figure(figsize=(9,9))

plt.scatter(x,y, color='purple', alpha=0.3, label='patients')

plt.plot(x, y_head, color='blue', alpha=0.6, label='linear')

plt.plot(x, y_head2, color='green', alpha=0.6, label='poly')

plt.xlabel('Pelvic Incidence Values')

plt.ylabel('Sacral Slope Values')

plt.legend()

plt.show()
# Polynomial Linear Regression Prediction:

print('Sacral Slope Value of the Patient with 100 Pelvic Incidence Value: ', lr2.predict([[1,110,12100]]))



# Linear Regression Prediction:

print('Sacral Slope Value of the Patient with 100 Pelvic Incidence Value: ', lr.predict([[110]]))
from sklearn.metrics import r2_score

print('Our linear models r square score is: ', r2_score(y, y_head))

print('Our polynomial models r square score is: ', r2_score(y, y_head2))
from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor()



tree.fit(x,y)



x_tree = np.arange(min(x), max(x),0.1).reshape(-1,1)

y_tree = tree.predict(x_tree)
plt.figure(figsize=(9,9))

plt.scatter(x,y, color='purple', alpha=0.4, label='patients')

plt.plot(x, y_head, color='blue', alpha=0.6, label='linear') # Linear regression model fitted line

plt.plot(x, y_head2, color='green', alpha=0.6, label='poly') # Polynomial linear regression model fitted line

plt.plot(x_tree, y_tree, color='orange', alpha=0.5, label='decision tree') # Decision tree regression model fitted line

plt.xlabel('Pelvic Incidence Values')

plt.ylabel('Sacral Slope Values')

plt.legend()

plt.show()
# Linear Regression Prediction:

print('Sacral Slope Value of the Patient with 100 Pelvic Incidence Value: ', lr.predict([[110]]))



# Polynomial Linear Regression Prediction:

print('Sacral Slope Value of the Patient with 100 Pelvic Incidence Value: ', lr2.predict([[1,110,12100]]))



# Decision Tree Linear Regression Prediction:

print('Sacral Slope Value of the Patient with 100 Pelvic Incidence Value: ', tree.predict([[110]]))
from sklearn.ensemble import RandomForestRegressor



forest = RandomForestRegressor(n_estimators=100, random_state=42)



forest.fit(x,y)



x_for = np.arange(min(x), max(x),0.1).reshape(-1,1)

y_for = forest.predict(x_for)
plt.figure(figsize=(15,15))

plt.subplot(2,2,1)

plt.scatter(x,y, color='purple', alpha=0.3, label='patients')

plt.plot(x, y_head, color='blue', alpha=0.6, label='linear')

plt.xlabel('Pelvic Incidence Values')

plt.ylabel('Sacral Slope Values')

plt.legend()



plt.subplot(2,2,2)

plt.scatter(x,y, color='purple', alpha=0.3, label='patients')

plt.plot(x, y_head2, color='green', alpha=0.6, label='poly')

plt.xlabel('Pelvic Incidence Values')

plt.ylabel('Sacral Slope Values')

plt.legend()



plt.subplot(2,2,3)

plt.scatter(x,y, color='purple', alpha=0.3, label='patients')

plt.plot(x_tree, y_tree, color='orange', alpha=0.6, label='decision tree')

plt.xlabel('Pelvic Incidence Values')

plt.ylabel('Sacral Slope Values')

plt.legend()



plt.subplot(2,2,4)

plt.scatter(x,y, color='purple', alpha=0.3, label='patients')

plt.plot(x_for, y_for, color='red', alpha=0.6, label='random forest')

plt.xlabel('Pelvic Incidence Values')

plt.ylabel('Sacral Slope Values')

plt.legend()

plt.show()
# Linear Regression Prediction:

print('Sacral Slope Value of the Patient with 100 Pelvic Incidence Value: ', lr.predict([[110]]))



# Polynomial Linear Regression Prediction:

print('Sacral Slope Value of the Patient with 100 Pelvic Incidence Value: ', lr2.predict([[1,110,12100]]))



# Decision Tree Linear Regression Prediction:

print('Sacral Slope Value of the Patient with 100 Pelvic Incidence Value: ', tree.predict([[110]]))



# Random Forest Regression Prediction:

print('Sacral Slope Value of the Patient with 100 Pelvic Incidence Value: ', forest.predict([[110]]))