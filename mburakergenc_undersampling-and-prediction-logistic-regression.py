import pandas as pd

import numpy as np

import datetime

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from imblearn.under_sampling import RandomUnderSampler

from __future__ import division

from sklearn import cross_validation

from sklearn.metrics import confusion_matrix,recall_score,precision_recall_curve,auc,roc_curve,roc_auc_score,classification_report
data = pd.read_csv("../input/creditcard.csv")
data.head()
data.info()
data.isnull().sum()
classes = pd.value_counts(data["Class"], sort=True).sort_index()

print(classes)
classes.plot(kind='bar', figsize=(12,7))

plt.title("Fraud Class Histogram")

plt.xlabel("Class")

plt.ylabel("Frequency")

plt.show()
normal_transactions = len(data[data["Class"] == 0])

fraud_transactions = len(data[data["Class"] == 1])

total_transactions = normal_transactions + fraud_transactions
print(p_normal_transactions)

print(p_fraud_transactions)
p_normal_transactions = (normal_transactions / total_transactions) * 100

p_fraud_transactions = (fraud_transactions / total_transactions) * 100
normal_trcs = data[data['Class'] == 0]

fraud_trcs = data[data['Class'] == 1]

plt.figure(figsize=(10,6))

plt.subplot(121)

normal_trcs.Amount.plot.hist(title="Fraud Transactions")

plt.subplot(122)

fraud_trcs.Amount.plot.hist(title="Normal Transactions")

plt.show()
def convertsecstohours(seconds):

    return datetime.datetime.fromtimestamp(seconds)



time_analyisis = data[['Time', 'Amount', 'Class']].copy()

time_analyisis['datetime'] = time_analyisis.Time.apply(convertsecstohours)

time_analyisis['hour'] = time_analyisis.datetime.dt.hour

cp_time_analysis = time_analyisis.groupby(['Class','hour'])['Amount'].count()
# Create an undersampler object

rus = RandomUnderSampler(return_indices=True)
# Drop the time and amount features

new_data = data.drop(data.columns[[0, 30]], axis=1)

X = new_data.values

y = data.Class.values
# Resample the features for training data and the target

X_resampled, y_resampled, idx_resampled = rus.fit_sample(X, y)
# Revert resampeled data into a dataframe

X_resampled = pd.DataFrame(X_resampled)

y_resampled = pd.DataFrame(y_resampled)

y_resampled.columns = ['Class']

undersampled_data = pd.concat([X_resampled, y_resampled], axis=1)
# Split the resampeled data into training and test sets

X = undersampled_data.values

y = undersampled_data.Class.values

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y ,test_size=0.2)
lr_model = LogisticRegression()

lr_model.fit(X_train, y_train)
# Test the model using the test set

predictions = lr_model.predict(X_test)
# Let's see the confusion matrix and evaluate the model 

cnf_matrix=confusion_matrix(y_test,predictions)
print("the recall for this model is :",cnf_matrix[1,1]/(cnf_matrix[1,1]+cnf_matrix[1,0]))
fig= plt.figure(figsize=(6,3))# to plot the graph

print("TP",cnf_matrix[1,1,]) # no of fraud transaction which are predicted fraud

print("TN",cnf_matrix[0,0]) # no. of normal transaction which are predited normal

print("FP",cnf_matrix[0,1]) # no of normal transaction which are predicted fraud

print("FN",cnf_matrix[1,0]) # no of fraud Transaction which are predicted normal

sns.heatmap(cnf_matrix,cmap="coolwarm_r",annot=True,linewidths=0.5)

plt.title("Confusion_matrix")

plt.xlabel("Predicted_class")

plt.ylabel("Real class")

plt.show()

print("\n----------Classification Report------------------------------------")

print(classification_report(y_test,predictions))