import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score



sns.set()

%matplotlib inline
df = pd.read_csv("../input/salary-data/Salary_Data.csv")
df.head()
df.shape
df.isnull().values.any()
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
df_copy = train_set.copy()
df_copy.shape
df_copy.head()
df_copy.describe()
df_copy.corr()
df_copy.plot.scatter(x='YearsExperience', y='Salary')
# Regression plot

sns.regplot('YearsExperience', # Horizontal axis

           'Salary', # Vertical axis

           data=df_copy)
test_set_full = test_set.copy()



test_set = test_set.drop(["Salary"], axis=1)
test_set.head()
train_labels = train_set["Salary"]
train_labels.head()
train_set_full = train_set.copy()



train_set = train_set.drop(["Salary"], axis=1)
train_set.head()
lin_reg = LinearRegression()



lin_reg.fit(train_set, train_labels)
salary_pred = lin_reg.predict(test_set)



salary_pred
print("Coefficients: ", lin_reg.coef_)

print("Intercept: ", lin_reg.intercept_)
print(salary_pred)

print(test_set_full["Salary"])
lin_reg.score(test_set, test_set_full["Salary"])
r2_score(test_set_full["Salary"], salary_pred)
plt.scatter(test_set_full["YearsExperience"], test_set_full["Salary"], color='blue')

plt.plot(test_set_full["YearsExperience"], salary_pred, color='red', linewidth=2)
# imports

import pandas as pd

import matplotlib.pyplot as plt



# this allows plots to appear directly in the notebook

%matplotlib inline



# read data into a DataFrame

data = pd.read_csv('../input/salary-data/Salary_Data.csv', index_col=0)

data.head()
# print the shape of the DataFrame

data.shape
data.hist()

plt.show()
data.plot(kind='density', subplots=True, layout=(1,1), sharex=False)

plt.show()
data.plot(kind='box', subplots=True, layout=(1,1), sharex=False, sharey=False)

plt.show()
import numpy as np
import matplotlib.pyplot as plt



from pandas.plotting import scatter_matrix

scatter_matrix(data)

plt.show()