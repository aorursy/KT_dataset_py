# Import the necessary packages used in this notebook

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os

from sklearn.decomposition import PCA, TruncatedSVD



from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.graph_objects as go

import plotly

init_notebook_mode(connected=True) #do not miss this line



from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,recall_score,precision_recall_curve,average_precision_score

from sklearn.utils import resample

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
datafr = pd.read_csv("../input/creditcard.csv", error_bad_lines=False)
display(datafr.shape)
display(datafr.head(10))
display(datafr.tail(10))
hours = (datafr['Time']/3600).astype(int)

datafr['Hours'] = hours



days = (datafr['Time']/86400).astype(int)

datafr['Days'] = days
bins = [0,100,1000,5000,10000,20000, 30000]

labels = [1,2,3,4,5,6]

datafr['binned'] = pd.cut(datafr['Amount'], bins=bins, labels=labels)

datafr.head(10)
f, axes = plt.subplots(1, 2, sharey=True, figsize=(15, 8))

sns.boxplot(x="binned", y="Amount", hue="Class", data=datafr[datafr['Class']==0], palette='Blues', ax=axes[0])

axes[0].set_title('BoxPlot for {}'.format("Class 0: Not Fraudulent"))

sns.boxplot(x="binned", y="Amount", hue="Class", data=datafr[datafr['Class']==1], palette='Purples', ax=axes[1])

axes[1].set_title('BoxPlot for {}'.format("Class 1: Fraudulent"))
plt.figure(figsize=(14,6))

sns.set(style="darkgrid")

sns.countplot(x='binned',data = datafr, hue = 'Class',palette='BuPu')

plt.title("Count Plot of Transactions per each amount bin\n", fontsize=16)

sns.set_context("paper", font_scale=1.4)

plt.show()
plt.figure(figsize=(14,6))

sns.set(style="darkgrid")

sns.countplot(x='Hours',data = datafr, hue = 'Class',palette='BuPu')

plt.title("Count Plot of Transactions per each Hour\n", fontsize=16)

sns.set_context("paper", font_scale=1.4)

plt.show()
print("Fraudulent Transactions:", len(datafr[datafr['Class']==1]))

print("Usual Transactions:", len(datafr[datafr['Class']==0]))
fraud =len(datafr[datafr['Class']==1])

notfraud = len(datafr[datafr['Class']==0])



# Data to plot

labels = 'Fraud','Not Fraud'

sizes = [fraud,notfraud]



# Plot

plt.figure(figsize=(7,6))

plt.pie(sizes, explode=(0.1, 0.1), labels=labels, colors=sns.color_palette("BuPu"),

autopct='%1.1f%%', shadow=True, startangle=0)

plt.title('Pie Chart Ratio of Transactions by their Class\n', fontsize=16)

sns.set_context("paper", font_scale=1.2)
y = datafr['Class']

X = datafr.drop(['Time','Class', 'binned'], axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)



model = XGBClassifier()

model.fit(X_train, y_train)



y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Test Accuracy is {:.2f}%".format(accuracy * 100.0))
model = XGBClassifier()

model.fit(X_train[['V1','V2','V3']], y_train)



y_pred = model.predict(X_test[['V1','V2','V3']])

accuracy = accuracy_score(y_test, y_pred)

print("Test Accuracy is {:.2f}%".format(accuracy * 100.0))
# assign cnf_matrix with result of confusion_matrix array

cnf_matrix = confusion_matrix(y_test,y_pred)

#create a heat map

sns.heatmap(pd.DataFrame(cnf_matrix), annot = True, cmap = 'Blues', fmt = 'd')

plt.xlabel('Predicted')

plt.ylabel('Expected')

plt.show()
from sklearn.utils import resample

# Separate input features and target

Y = datafr.Class

X = datafr.drop(['Time','Class','binned'], axis=1)



# setting up testing and training sets

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=2727)



# concatenate our training data back together

X = pd.concat([X_train, Y_train], axis=1)
# separate minority and majority classes

not_fraud = X[X.Class==0]

fraud = X[X.Class==1]



# upsample minority

fraud_upsampled = resample(fraud,

                          replace=True, # sample with replacement

                          n_samples=len(not_fraud), # match number in majority class

                          random_state=2727) # reproducible results



# combine majority and oversampled minority

oversampled = pd.concat([not_fraud, fraud_upsampled])



# check new class counts

oversampled.Class.value_counts()
# trying xgboost again with the balanced dataset

y_train = oversampled.Class

X_train = oversampled.drop('Class', axis=1)



upsampled = XGBClassifier()

upsampled.fit(X_train, y_train)



# Predict on test

upsampled_pred = upsampled.predict(X_test)



# predict probabilities

probs = upsampled.predict_proba(X_test)

# keep probabilities for the positive outcome only

probs = probs[:, 1]
# Checking accuracy

accuracy = accuracy_score(Y_test, upsampled_pred)

print("Test Accuracy is {:.2f}%".format(accuracy * 100.0))
# f1 score

f1_over = f1_score(Y_test, upsampled_pred)

print("F1 Score is {:.2f}%".format(f1_over))
from sklearn.metrics import auc

# calculate precision-recall curve

precision, recall, thresholds = precision_recall_curve(Y_test, probs)

# calculate precision-recall AUC

auc_over = auc(recall, precision)

# plot no skill

plt.plot([0, 1], [0.5, 0.5], linestyle='--')

# plot the precision-recall curve for the model

plt.plot(recall, precision, marker='.')

plt.title("Precison-Recall Curve for XGBoost with AUC score: {:.3f}".format(auc_over))

# show the plot

plt.show()
# assign cnf_matrix with result of confusion_matrix array

cnf_matrix = confusion_matrix(Y_test,upsampled_pred)

#create a heat map

sns.heatmap(pd.DataFrame(cnf_matrix), annot = True, cmap = 'Blues', fmt = 'd')

plt.xlabel('Predicted')

plt.ylabel('Expected')

plt.show()
# still using our separated classes fraud and not_fraud from above



# downsample majority

not_fraud_downsampled = resample(not_fraud,

                                replace = False, # sample without replacement

                                n_samples = len(fraud), # match minority n

                                random_state = 27) # reproducible results



# combine minority and downsampled majority

downsampled = pd.concat([not_fraud_downsampled, fraud])



# checking counts

downsampled.Class.value_counts()
# trying xgboost again with the balanced dataset

y_train = downsampled.Class

X_train = downsampled.drop('Class', axis=1)



undersampled = XGBClassifier()

undersampled.fit(X_train, y_train)



# Predict on test

undersampled_pred = undersampled.predict(X_test)

# predict probabilities

probs = undersampled.predict_proba(X_test)

# keep probabilities for the positive outcome only

probs = probs[:, 1]
# Checking accuracy

accuracy = accuracy_score(Y_test, undersampled_pred)

print("Test Accuracy is {:.2f}%".format(accuracy * 100.0))
# f1 score

f1_under = f1_score(Y_test, undersampled_pred)

print("F1 Score is {:.2f}%".format(f1_under))
from sklearn.metrics import auc

# calculate precision-recall curve

precision, recall, thresholds = precision_recall_curve(Y_test, probs)

# calculate precision-recall AUC

auc_under = auc(recall, precision)

# plot no skill

plt.plot([0, 1], [0.5, 0.5], linestyle='--')

# plot the precision-recall curve for the model

plt.plot(recall, precision, marker='.')

plt.title("Precison-Recall Curve for XGBoost with AUC score: {:.3f}".format(auc_under))

# show the plot

plt.show()
# assign cnf_matrix with result of confusion_matrix array

cnf_matrix = confusion_matrix(Y_test,undersampled_pred)

#create a heat map

sns.heatmap(pd.DataFrame(cnf_matrix), annot = True, cmap = 'Blues', fmt = 'd')

plt.xlabel('Predicted')

plt.ylabel('Expected')

plt.show()
from imblearn.over_sampling import SMOTE



# Separate input features and target

Y = datafr.Class

X = datafr.drop(['Time','Class','binned'], axis=1)



# setting up testing and training sets

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=2727)



sm = SMOTE(random_state=2727, ratio=1.0)

X_train, Y_train = sm.fit_sample(X_train, Y_train)
X_train = pd.DataFrame(data=X_train)

X_train.columns = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11',

       'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21',

       'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount', 'Hours', 'Days']

Y_train = pd.Series(Y_train)
smote = XGBClassifier()

smote.fit(X_train, Y_train)



# Predict on test

smote_pred = smote.predict(X_test)

# predict probabilities

probs = smote.predict_proba(X_test)

# keep probabilities for the positive outcome only

probs = probs[:, 1]
# Checking accuracy

accuracy = accuracy_score(Y_test, smote_pred)

print("Test Accuracy is {:.2f}%".format(accuracy * 100.0))
# f1 score

f1_smote = f1_score(Y_test, smote_pred)

print("F1 Score is {:.2f}%".format(f1_smote))
from sklearn.metrics import auc

# calculate precision-recall curve

precision, recall, thresholds = precision_recall_curve(Y_test, probs)

# calculate precision-recall AUC

auc_smote = auc(recall, precision)

# plot no skill

plt.plot([0, 1], [0.5, 0.5], linestyle='--')

# plot the precision-recall curve for the model

plt.plot(recall, precision, marker='.')

plt.title("Precison-Recall Curve for XGBoost with AUC score: {:.3f}".format(auc_smote))

# show the plot

plt.show()
# assign cnf_matrix with result of confusion_matrix array

cnf_matrix = confusion_matrix(Y_test,smote_pred)

#create a heat map

sns.heatmap(pd.DataFrame(cnf_matrix), annot = True, cmap = 'Blues', fmt = 'd')

plt.xlabel('Predicted')

plt.ylabel('Expected')

plt.show()
# F1 Score list for all models

f1 = [f1_over, f1_under, f1_smote]

# AUC Score list for all models

auc = [auc_over, auc_under, auc_smote]

# Name List of ML Models used

models = ['Over-Sampling', 'Under-Sampling', 'SMOTE']

y_pos = np.arange(len(models)) #Position = 0,1,2



# Plot F1 Score

plt.figure(figsize=(10, 6))  

plt.bar(y_pos, f1, align='center', alpha=0.8, color=sns.color_palette("PuBu"))

plt.xticks(y_pos, models)

plt.ylabel('F1 Score')

plt.title('Performance based on F1 Score')



# Plot AUC Score

plt.figure(figsize=(10, 6))  

plt.bar(y_pos, auc, align='center', alpha=0.8, color=sns.color_palette("PuBu"))

plt.xticks(y_pos, models)

plt.ylabel('AUC Score')

plt.title('Performance based on AUC Score')