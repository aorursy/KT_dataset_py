# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import linear_model

import math



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
Original_training_data= pd.read_csv('../input/random-linear-regression/test.csv')

Original_test_data=pd.read_csv('../input/random-linear-regression/train.csv')



Original_training_data.info()

Original_training_data.describe()
Original_test_data.info()

Original_test_data.describe()
Clean_training_data=Original_training_data.dropna()

Clean_training_data.count()
Clean_testing_data=Original_test_data.dropna()

Clean_testing_data.count()
x_training_set = Clean_training_data.as_matrix(['x'])

y_training_set = Clean_training_data.as_matrix(['y'])



x_test_set = Clean_training_data.as_matrix(['x'])

y_test_set = Clean_training_data.as_matrix(['y'])
# So let's plot some of the data 

# - this gives some core routines to experiment with different parameters

plt.title('Relationship between X and Y')

plt.scatter(x_training_set, y_training_set,  color='black')

plt.show()



# Use subplot to have graphs side by side

plt.subplot(1, 2, 1)

plt.title('X training set')

plt.hist(x_training_set)



plt.subplot(1, 2, 2)

plt.title('Y training set')

plt.hist(y_training_set)

plt.show()



plt.subplot(1, 2, 1)

plt.title('X training set')

plt.boxplot(x_training_set)



plt.subplot(1, 2, 2)

plt.title('Y training set')

plt.boxplot(y_training_set)

plt.show()
plt.title('Relationship between X and Y')

plt.scatter(x_training_set, y_training_set,  color='black')

plt.show()
fig = plt.figure(figsize = (80,40))

ax=sns.barplot(x='x',y='y',data=Clean_testing_data)

var = ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
sns.lineplot(x='x',y='y',data=Clean_training_data)
# Now to set up the linear regression model

# Create linear regression object

lm = linear_model.LinearRegression()

# ... then fir it

lm.fit(x_training_set,y_training_set)



# Have a look at R sq to give an idea of the fit 

print('R sq: ',lm.score(x_training_set,y_training_set))



# and so the correlation is..

print('Correlation: ', math.sqrt(lm.score(x_training_set,y_training_set)))
# So let's run the model against the test data

y_predicted = lm.predict(x_test_set)



plt.title('Comparison of Y values in test and the Predicted values')

plt.ylabel('Test Set')

plt.xlabel('Predicted values')



plt.scatter(y_predicted, y_test_set,  color='blue')

plt.show()