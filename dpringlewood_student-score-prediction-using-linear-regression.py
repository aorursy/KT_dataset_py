# Handle all my imports

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

import seaborn as sns
# Import the data and inspect it

maths = pd.read_csv('../input/student-grade-prediction/student-mat.csv')

print(maths.info())

maths.head()

# Lets do some visual-EDA on our data which we will use



fig, axes = plt.subplots(2, 2, figsize=(16,12))



sns.regplot('G2', 'G3', data=maths, ax=axes[0, 0]).set_title('G2 vs G3 grades')

sns.swarmplot('failures', 'G3', data=maths, ax=axes[1, 0]).set_title('Effect of Failures on Final Grade')

sns.swarmplot('famrel', 'G3', data=maths, ax=axes[0, 1]).set_title('Effect of Family Relationships on Final Grade')

sns.swarmplot('studytime', 'G3', data=maths, ax=axes[1, 1]).set_title('Effect of Studytime on Final Grade')

plt.tight_layout()

plt.show()
maths = maths.select_dtypes('int64')

maths = maths[['famrel', 'studytime', 'failures', 'absences', 'G1', 'G2', 'G3']]

print(maths.info())



# set our prediction of a students final score (G3)

predict = 'G3'



# split-up X & y and make sure that they are np array's

# sklearn needs numpy array's as inputs

X = np.array(maths.drop(predict, axis=1))

y = np.array(maths[predict])
# split-up our current X & y variables into training

# and testing data.



X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.8, random_state=42

)



# Now we need to initiate our model and train it



linear = LinearRegression()

linear.fit(X_train, y_train)
# Lets take a look at how well this model preforms

print("The R^2 is: ", linear.score(X_test, y_test))

coeff = linear.coef_

intercept = linear.intercept_



for i in range(len(coeff)):

    print(maths.columns[i], ': ', coeff[i])

print('The intercept of our slope is: ', intercept)