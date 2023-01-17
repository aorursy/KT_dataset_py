# import libraries to load and visualize data

import pandas as pd

import matplotlib.pyplot as  plt
# load the data

df = pd.read_csv('/kaggle/input/hr-analytics/HR_comma_sep.csv')

df
# check if there's any NA values

df.isnull().values.any()
# find data types of our dataset/dataFrame

df.dtypes
# salary has ordinal value (has order between values)

cleanup_vals = {"salary": {'low': 1,

                           'medium': 2,

                           'high': 3}}

df.replace(cleanup_vals, inplace=True)

df
# one hot encoding for Department column

dummies = pd.get_dummies(df.Department)

dummies
# merge dummies with df

merged = pd.concat([df, dummies], axis='columns')

merged
# delete unnecessary columns

# Department column and one dummy variable (to avoid dummy variable trap)

final_df = merged.drop(['Department', 'technical'], axis='columns')

final_df
# plot data to have an idea of which algorithm we might need

plt.scatter(x=final_df['salary'], y=final_df['left'])
plt.scatter(x=final_df['satisfaction_level'], y=final_df['left'])
plt.scatter(x=final_df['time_spend_company'], y=final_df['left'])
# we can go for logistic regression algorithm.

# this is a classification problem, cuz we need to decide an employee will

#  stay or not.

from sklearn.linear_model import LogisticRegression
# get X, y data for training

X = final_df.drop(['left'], axis='columns')

y = final_df.left
# split 10% data for testing

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
# train the model

model = LogisticRegression()

model.fit(X_train, y_train)
# test accuracy

model.score(X_test, y_test)
model.predict([[0.85, 0.87, 6, 232, 5, 0, 0, 3, 0, 0, 0,0,1,0,0,0,0]])