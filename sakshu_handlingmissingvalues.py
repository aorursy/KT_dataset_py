import numpy as np # linear algebra

import pandas as pd # data processing

import matplotlib.pyplot as plt
data = pd.read_csv('../input/pima-indians-diabetes.csv')

data.head()
cols = ['Glucose', 'BP', 'Skin', 'Insulin', 'BMI', 'Pedigree', 'Age']

dataset =data[cols].replace(0, np.nan)

dataset.info()
missing_df = dataset

missing_df.replace(0, np.nan, inplace=True)

missing_df = missing_df.isnull().sum(axis=0).reset_index()

missing_df.columns = ['column_name', 'missing_count']

missing_df = missing_df.loc[missing_df['missing_count']>0]

missing_df = missing_df.sort_values(by='missing_count')



ind = np.arange(missing_df.shape[0])

fig, ax = plt.subplots()

rects = ax.barh(ind, missing_df.missing_count.values, color='blue')

ax.set_yticks(ind)

ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')

ax.set_xlabel("Count of missing values")

ax.set_title("Number of missing values in each column")

plt.show()
data['Skin'].replace(0, np.nan, inplace=True)

data.head()
data_without_missing_values = data.dropna()

data_without_missing_values.head()
data_with_missing_values = data[data.isnull().any(axis=1)]

data_with_missing_values.head()
X_train = data_without_missing_values.drop(['Skin'], axis=1)

X_train.head()
y_train = data_without_missing_values['Skin']

y_train.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.25, random_state = 10)

X_train.shape
from sklearn.linear_model import BayesianRidge

model = BayesianRidge(compute_score = True, fit_intercept= True, tol= 10)

model = model.fit(X_train, y_train)
print("Results For Bayesian Ridge...Before Filling Null Values")

scoreBeforeFilling = model.score(X_test, y_test)

print("\nScore", scoreBeforeFilling*100)
X_test_nan = data_with_missing_values.drop(['Skin'], axis=1)

y_test_nan = data_with_missing_values['Skin']
predicted = model.predict(X_test_nan)

predicted =predicted.astype(int)
data_with_missing_values['Skin'] = predicted
data_after_filling = pd.concat([data_without_missing_values, data_with_missing_values], ignore_index = True)
from sklearn.utils import shuffle

data_after_filling = shuffle(data_after_filling, random_state=10)
X_train = data_after_filling.drop(['Skin'], axis=1)

y_train = data_after_filling['Skin']
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.25, random_state = 10)
from sklearn.linear_model import BayesianRidge

model = BayesianRidge(compute_score = True, fit_intercept= True, tol= 10)

model = model.fit(X_train, y_train)
print("Results For Bayesian Ridge... After Filling Null Values")

scoreAfterFilling = model.score(X_test, y_test)

print("\nScore", scoreAfterFilling*100)