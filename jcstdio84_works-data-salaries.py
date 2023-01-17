#=======Preprocessing data part==============



# Importing the libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns





# Importing the dataset

dataset = pd.read_csv('../input/Works_Data.csv', sep=';', header=0)



# Spliting into Independent and dependent variables

X = dataset.iloc[:, 1:2].values

y = dataset.iloc[:, -1].values



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



# Fitting Simple Linear Regression to the Training set

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)



# Predicting the Test set results

y_pred = regressor.predict(X_test)



# Visualising the Training set results

plt.scatter(X_train, y_train, color = 'red')

plt.plot(X_train, regressor.predict(X_train), color = 'blue')

plt.title('Salary vs Years of Experience (Training set)')

plt.xlabel('Years of Experience')

plt.ylabel('Salary')

plt.show()



# Visualising the Test set results

plt.scatter(X_test, y_test, color = 'red')

plt.plot(X_train, regressor.predict(X_train), color = 'blue')

plt.title('Salary vs Experience (Test set)')

plt.xlabel('Years of Experience')

plt.ylabel('Salary')

plt.show()





#=================Visualizing the data====================



# Importing the dataset

dataset = pd.read_csv('../input/Works_Data.csv', sep=';', header=0)



# Spliting into Independent and dependent variables

X = dataset.iloc[:, 0:-1].values

y = dataset.iloc[:, -1].values

# Actvate the color set

sns.set(color_codes=True)





# Categorical plot of salaries distributed by cities

# Section under revision by an error: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()

# Working to solve it

# ax_cat = sns.catplot(x=X[:, 0], y=y, kind='boxen', data=dataset.sort_values('Salario'))

# ax_cat.set_xticklabels(ax_cat.get_xticklabels(), rotation = 40, ha = "right")







# Plot showing the count of works per province

fig, ax = plt.subplots(figsize=(16,5))

sns.countplot(X[:, 0], ax=ax)

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")



from scipy import stats



print("""

      Most paid job:  {}

      Less paid job:  {}

      Average salary: {}

      Mode salary:    {}



""".format(max(y), min(y), np.mean(y), int(stats.mode(y)[0])))