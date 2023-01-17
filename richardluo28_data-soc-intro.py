from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = 'all'
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/iris.csv')

df.head()
df.describe()
df.columns

df.species.describe()

df.species.unique()
import matplotlib.pyplot as plt

# The following line just for the convience of seeing the plot in the notebook sheet

%matplotlib inline
from random import randint, seed

seed(20)

x = [i for i in range(1, 21)]

y = [randint(20, 60) for _ in range(20)]

y_x = list(zip(x, y))

print(y_x)

plt.xticks(np.arange(0, 21, 1));

plt.plot(x,y);
plt.bar(x, y);



plt.ylabel('people')

plt.xlabel('language')

plt.bar(['Java', 'c++', 'c', 'python', 'R'], [10, 80, 70, 80, 50], color = 'orange');
column_name = df.columns

print(column_name)

df[column_name[0]].plot(legend = True, xticks = np.arange(0, 151, 10));

df[column_name[1]].plot(legend = True, xticks = np.arange(0, 151, 10));

df.plot(xticks = np.arange(0, 151, 10));

df.species.unique()
setosa_df = df.query('species == "setosa"')

versicolor_df = df.query('species == "versicolor"')

virginica_df = df.query('species == "virginica"')

setosa_df.describe()

versicolor_df.describe()

virginica_df.describe()
setosa_df = df[df.species == 'setosa']

versicolor_df = df[df.species == 'versicolor']

virginica_df = df[df.species == 'virginica']

setosa_df.describe()

versicolor_df.describe()

virginica_df.describe()
setosa_df.plot(yticks=np.arange(0, 9, 1))

versicolor_df.plot(use_index = False, yticks=np.arange(0, 9, 1))

virginica_df.plot(use_index = False, yticks=np.arange(0, 9, 1))
# Plot a scatter chart using x='sepal_length', y='sepal_width', and separate colors for each of the three dataframes

ax = setosa_df.plot.scatter(x='sepal_length', y='sepal_width', label='setosa')

ax = versicolor_df.plot.scatter(x='sepal_length', y='sepal_width', label='versicolor', color='green', ax=ax)

ax = virginica_df.plot.scatter(x='sepal_length', y='sepal_width', label='virginica', color='red', ax=ax)
# Plot a scatter chart using x='sepal_length', y='sepal_width', and separate colors for each of the three dataframes

ax = setosa_df.plot.scatter(x='petal_length', y='petal_width', label='setosa')

ax = versicolor_df.plot.scatter(x='petal_length', y='petal_width', label='versicolor', color='green', ax=ax)

ax = virginica_df.plot.scatter(x='petal_length', y='petal_width', label='virginica', color='red', ax=ax)
df2 = df.copy()

df2.describe()

df.describe()
df2.drop(['sepal_length', 'sepal_width'], inplace = True, axis = 1)

# axis = 1 is column, axis = 0 is row

df2.describe()
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
y = df2.pop('species')

X = df2
X.head()

y.head()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)

X_train.head()

X_train.describe()

y_train.head()
# define a model

clf = LogisticRegression(random_state=0, solver='lbfgs',

                         multi_class='multinomial')

# fit the data, training

clf.fit(X_train, y_train)
# predict the data base on the 

y_predict = clf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score([1, 1, 0], [1, 1, 1])
accuracy_score(y_test, y_predict)
# can also use the model score function

clf.score(X_test, y_test)
y = df.pop('species')
X = df

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)



clf2 = LogisticRegression(random_state=0, solver='lbfgs',

                         multi_class='multinomial')

# fit the data, training

clf2.fit(X_train, y_train)

clf2.score(X_test, y_test)
X = df

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



clf2 = LogisticRegression(random_state=0, solver='lbfgs',

                         multi_class='multinomial')

# fit the data, training

clf2.fit(X_train, y_train)

clf2.score(X_test, y_test)
import pickle
pickle.dump(clf2, open('logs_model.pkl', 'wb'))
loaded_model = pickle.load(open('logs_model.pkl', 'rb'))

result = loaded_model.score(X_test, y_test)

result