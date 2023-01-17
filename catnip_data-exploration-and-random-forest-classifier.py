import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas_profiling
raw_data = pd.read_csv('/kaggle/input/personal-loan-modeling/Bank_Personal_Loan_Modelling.csv')
raw_data.profile_report()
from sklearn.model_selection import train_test_split

X = raw_data.drop(['Personal Loan', 'ID'], axis=1)

Y = raw_data['Personal Loan']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2)
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

model.fit(X_train, Y_train)
predictions = model.predict(X_test)

import seaborn as sns

from matplotlib import pyplot as plt

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

ac = accuracy_score(Y_test, predictions)

precision = precision_score(Y_test, predictions)

recall = recall_score(Y_test, predictions)

print(f'Accuracy: {ac:.3}\nPrecision:{precision:.3}\nRecall: {recall:.3}')

cm = confusion_matrix(Y_test, predictions)

labels = ['No', 'Yes']

sns.heatmap(cm, xticklabels=labels, yticklabels=labels, square=True, annot=True, fmt="d")

plt.title("Confusion Matrix", fontsize=24)

plt.ylabel('Ground truth', fontsize=16)

plt.xlabel('Prediction', fontsize=16)

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)
feature_dict = {

    'feature': X_train.columns,

    'importance': model.feature_importances_

}

feature_df = pd.DataFrame(feature_dict)

feature_df.sort_values(by='importance', ascending=False)