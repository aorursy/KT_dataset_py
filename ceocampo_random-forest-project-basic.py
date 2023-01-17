# Import libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

%matplotlib inline
import pandas as pd

loans = pd.read_csv("../input/loan_data.csv")

loans.head()



# Data contains information on potential clients of LendingClub.com from 2007-2010.

# Goal is to classify and predict if a borrower will pay his or her loan back in full based on these given features.
loans.info()
# Check for missing values

loans.isnull().sum()
loans.describe()
sns.distplot(loans['fico'])

plt.show()
plt.figure(figsize=(10,6))

loans[loans['credit.policy'] == 1]['fico'].hist(bins=30, alpha=0.6, label='Credit.Policy=1')

loans[loans['credit.policy'] == 0]['fico'].hist(bins=30, alpha=0.6, label='Credit.Policy=0')

plt.xlabel('FICO')

plt.legend()

plt.show()
plt.figure(figsize=(10,6))

loans[loans['not.fully.paid'] == 1]['fico'].hist(bins=30, alpha=0.6, label='not.fully.paid=1')

loans[loans['not.fully.paid'] == 0]['fico'].hist(bins=30, alpha=0.6, label='not.fully.paid=0')

plt.xlabel('FICO')

plt.legend()

plt.show()
plt.figure(figsize=(12,6))

sns.countplot(x='purpose', hue='not.fully.paid', data=loans)

plt.tight_layout

plt.show()
sns.jointplot(x='fico', y='int.rate', data=loans, space=0.2)

plt.show()
sns.lmplot(x='fico', y='int.rate', hue='credit.policy', data=loans, col='not.fully.paid')

plt.show()
loans['purpose'].nunique()
# Converting categorical variables to dummy variables

cat_feats = ['purpose']

final_data = pd.get_dummies(loans, columns=cat_feats, drop_first=True)

final_data.head()
# Spltting data into training and test data sets



from sklearn.model_selection import train_test_split

X = final_data.drop('not.fully.paid', axis=1)

y = final_data['not.fully.paid']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# We will use a single decision tree first before using a random forest model

from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()

dtree.fit(X_train, y_train)

y_pred_tree = dtree.predict(X_test)
# Decision Tree Model Evaluation

from sklearn.metrics import confusion_matrix, classification_report

print(classification_report(y_test, y_pred_tree))

print(confusion_matrix(y_test, y_pred_tree))
# Training a Random Forest Model

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=300)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
# Random Forest Model Evaluation

print(classification_report(y_test, y_pred_rf))

print(confusion_matrix(y_test, y_pred_rf))
loans.shape
# Random forest was slightly more accurate overall, which is expected with an ensemble of decision trees vs. one decision tree.

# If more data were to be collected and used to retreain each model, the ensemble model will most likey start to outperform the single decision tree more significantly.