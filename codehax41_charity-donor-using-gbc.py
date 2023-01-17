import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from IPython.display import display

%inline matplotlib
import os

print(os.listdir("../input"))
df_train = pd.read_csv('../input/udacity-mlcharity-competition/census.csv')

df_test = pd.read_csv('../input/udacity-mlcharity-competition/test_census.csv')
df_train.head()
n_records = df_train.shape[0]

n_greater_50k = len(df_train[df_train['income'] == '>50K'])

n_at_most_50k = len(df_train[df_train['income'] == '<=50K'])

greater_percent = 100 * n_greater_50k / n_records



print("Total number of records: ",n_records)

print("Individuals making more than $50,000: ",n_greater_50k)

print("Individuals making at most $50,000: ",n_at_most_50k)

print("Percentage of individuals making more than $50,000: ",greater_percent)
df_train[['capital-gain','capital-loss']].hist()
# Split the data into features and target label

income_raw = df_train['income']

features_raw = df_train.drop('income', axis = 1)



# Transform Skewed Continuous Features

skewed = ['capital-gain', 'capital-loss']

features_log_transformed = pd.DataFrame(data = features_raw)

features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))



#Normalizing Numerical Features

# Import sklearn.preprocessing.StandardScaler

from sklearn.preprocessing import MinMaxScaler



# Initialize a scaler, then apply it to the features

scaler = MinMaxScaler() # default=(0, 1)

numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']



features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)

features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])

# One-hot encode the 'features_log_minmax_transform' data using pandas.get_dummies()

features_final = pd.get_dummies(features_log_minmax_transform)



# Encode the 'income_raw' data to numerical values

income = income_raw.map({'<=50K': 0, '>50K':1})



# Print the final features

features_final.head(5)
# check correlation between features: 

import seaborn as sns

data = pd.concat([features_final, income], axis =1)

plt.figure(figsize=(30,28))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(data.corr(),linewidths=0.1,vmax=1.0, 

            square=True,linecolor='white')
# Import train_test_split

from sklearn.model_selection import train_test_split



# Split the 'features' and 'income' data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(features_final, income, test_size = 0.2, random_state = 21)



# Show the results of the split

print("Training set has {} samples",X_train.shape[0])

print("Testing set has {} samples.",X_test.shape[0])
# Import 'GridSearchCV', 'make_scorer', and any other necessary libraries

from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import AdaBoostClassifier
# Initialize the classifier

clf = AdaBoostClassifier(random_state=42)



# Create the parameters list you wish to tune, using a dictionary if needed.

parameters = {'n_estimators': [200, 300, 500, 600]}



# Make an roc_auc scoring object using make_scorer()

scorer = make_scorer(roc_auc_score)



# Perform grid search on the classifier using 'scorer' as the scoring method using GridSearchCV()

grid_obj = GridSearchCV(clf, parameters, scoring=scorer, cv=5)



# Fit the grid search object to the training data and find the optimal parameters using fit()

grid_fit = grid_obj.fit(X_train, y_train)



# Get the estimator

best_clf = grid_fit.best_estimator_



print("best_estimator", grid_fit.best_estimator_)

# Make predictions using the unoptimized and model

predictions = (clf.fit(X_train, y_train)).predict(X_test)

best_predictions = best_clf.predict(X_test)



# Report the before-and-afterscores

print("Unoptimized model\n------")

print("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions)))

print("Area under curve on testing data: {:.4f}".format(roc_auc_score(y_test, predictions)))

print("\nOptimized Model\n------")

print("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))

print("Final Area under curve on  the testing data: {:.4f}".format(roc_auc_score(y_test, best_predictions)))
from sklearn.preprocessing import MinMaxScaler

# Replace all NaNs with forwardfilling

for row in df_test:

    df_test[row].fillna(method='ffill', axis=0, inplace=True)

# Transform Skewed Continuous Features

skewed = ['capital-gain', 'capital-loss']

features_test_log_transformed = pd.DataFrame(data = df_test)

features_test_log_transformed[skewed] = features_test_log_transformed[skewed].apply(lambda x: np.log(x + 1))

#Normalizing Numerical Features

# Initialize a scaler, then apply it to the features

scaler = MinMaxScaler() # default=(0, 1)

numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']



features_test_log_minmax_transform = pd.DataFrame(data = features_test_log_transformed)

features_test_log_minmax_transform[numerical] = scaler.fit_transform(features_test_log_transformed[numerical])



# One-hot encode the 'features_log_minmax_transform' data using pandas.get_dummies()

features_test_encoded = pd.get_dummies(features_test_log_minmax_transform)



# Remove the first column

features_test_final = features_test_encoded.drop('Unnamed: 0',1)
# Make predictions using features_test_final and store it a new coulmn in test dataset

df_test['id'] = df_test.iloc[:,0]

df_test['income'] = best_clf.predict_proba(features_test_final)[:,1]

df_test.head()
# write output file

df_test[['id', 'income']].to_csv("submission.csv", index=False)