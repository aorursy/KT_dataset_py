# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
# Read the data into a pandas dataframe
df = pd.read_csv('../input/Admission_Predict_Ver1.1.csv')
# Store a copy of the original dataframe
df_with_predictions = df.copy()
# Eyeball the data
print(df.info())
df.head()
# Remove whitespaces in the headers
df.rename(columns=lambda x: x.strip(), inplace=True) 

# Drop the column labelled 'Serial No.'. It does not affect our analysis because it is a nominal value.
df.drop(columns={'Serial No.'}, inplace=True)
df.head()
# Display a description of the dataframe
df.describe()
# Eyeball the correlation between all columns using sns.pairplot
sns.pairplot(df, kind='scatter', hue='University Rating', palette='husl')
# Eyeball the correlation between all columns using sns.heatmap
fig, ax = plt.subplots(figsize=(10,8))
sns.heatmap(df.corr(), ax=ax, annot=True, linewidths=1, fmt='.2f',
            cmap="viridis", vmin=0, vmax=1)
plt.show()
df.corr()['Chance of Admit'].round(1)
df.drop(columns={'LOR', 'Research'}, inplace=True)
df.head()
df_preprocessed = df.copy()
df_preprocessed.head(10)
# Display the shape of the dataframe
df_preprocessed.shape
df['Chance of Admit'].median()
# Create the targets
targets = np.where(df['Chance of Admit'] >= df['Chance of Admit'].median(), 1, 0)
targets.shape
# Adding the column 'Probability of Acceptance' to our dataframe
df['Probability of Acceptance'] = targets
df.head()
targets.sum()/len(targets)
# create a checkpoint
# drop column 'Chance of Admit' to avoid multicollinearity
data_with_targets = df.drop(['Chance of Admit'], axis=1)
data_with_targets.head()
# Display the shape of the dataframe
data_with_targets.shape
# Select the inputs
inputs = data_with_targets.iloc[:, :-1]
inputs.head()
# Splitting the data into train and test sets with a 80-20 split percentage
x_train, x_test, y_train, y_test = train_test_split(inputs, targets, test_size=.2, shuffle=True, random_state=20)
# Displaying the train and test datasets and targets
print(x_train, y_train)
print(x_test, y_test)
# Displaying the shape of the train and test datasets and targets
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
# Create a logistic regression object
log_reg = LogisticRegression()
# Fit the data to the model
log_reg.fit(x_train, y_train)
# Display the accuracy of the model
log_reg.score(x_train, y_train)
# Accuracy means that x% of the model outputs match the targets
model_outputs = log_reg.predict(x_train)
model_outputs
y_train
model_outputs == y_train
# Model accuracy
np.sum(model_outputs == y_train) / len(model_outputs)
# Intercept value
log_reg.intercept_
# Coefficient value
log_reg.coef_
feature_name = inputs.columns.values
summary_table = pd.DataFrame(columns=['Feature Name'], data=feature_name)
summary_table['Coefficient'] = np.transpose(log_reg.coef_)
summary_table.head()
# Adding the intercept value to the summary table
summary_table.index += 1
summary_table.loc[0] = ['Intercept', log_reg.intercept_[0]]
summary_table.sort_index(inplace=True)
summary_table.head()
# Finding the odds ratio
summary_table['Odds_ratio'] = np.exp(summary_table.Coefficient)
summary_table
log_reg.score(x_test, y_test)
# Predict the probability of an output being 0 (first column) or 1 (second column)
predicted_proba = log_reg.predict_proba(x_test)
predicted_proba
# Shape of the test-data set
predicted_proba.shape
# Slice out the values from the second column
probability_admit = predicted_proba[:,1]
probability_admit
# Predicted values
pred = log_reg.predict(x_test)
pred
predicted_value = log_reg.predict(x_test)
predicted_value
df_with_predicted_outcomes = inputs.copy()

probability_admit = pd.DataFrame(probability_admit)
df_with_predicted_outcomes['Probability'] = probability_admit

predicted_value = pd.DataFrame(predicted_value)
df_with_predicted_outcomes['Prediction'] = predicted_value
# Display the final dataframe with predictions
df_with_predicted_outcomes.head(10)
