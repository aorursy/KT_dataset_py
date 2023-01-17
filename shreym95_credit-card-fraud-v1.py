import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import datetime

import time

import os

from pprint import pprint

from sklearn.preprocessing import MinMaxScaler

from sklearn.feature_selection import SelectKBest

from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier

from imblearn.over_sampling import SMOTE

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_recall_curve

from sklearn.metrics import mean_squared_error

from sklearn import model_selection

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report
!pwd
#Change directory to where dataset is - not required on kaggle

#os.chdir("X:\\Datasets\creditcardfraud")
#Reading the dataset

master_df=pd.read_csv('../input/creditcardfraud/creditcard.csv')
#Keeping our original dataset safe :)

transaction_data = master_df.copy()
display(transaction_data.head())

print("Dimensions of the dataset are:", transaction_data.shape)
#Data points for Fraudulent transactions

fraud = len(transaction_data[transaction_data["Class"]==1])

print("Total Fraud transactions are", fraud, ", which is") 

print(round(fraud/len(transaction_data) * 100, 2), "% of the dataset")
#Checking for null values (all values are numerical)

transaction_data.isnull().sum()



# We have no null values
#Let's look at the summary of the data

transaction_data.describe()
#Let's convert the time variable into hours from seconds before moving forward



transaction_data['Time'] = transaction_data['Time']/3600
#Distribution of Amount

figure, a = plt.subplots(1, 2, figsize=(15, 4))

amount = transaction_data['Amount'].values



sns.boxplot(amount, ax = a[0])

a[0].set_title('Distribution of Amount')



fraud_df = transaction_data[transaction_data['Class']==1]

amount_fraud = fraud_df['Amount'].values



sns.boxplot(amount_fraud, ax = a[1])

a[1].set_title('Distribution of Amount w.r.t. Fraud Transactions')

#Keeping a similar scale as previous graph for comparison

#a[1].set_xlim([min(amount), max(amount)])

plt.show()
#Checking distribution with time



figure, a = plt.subplots(1, 2, figsize=(15, 4))

time = transaction_data['Time'].values



sns.distplot(time, ax = a[0])

a[0].set_title('Distribution of Transactions with Time')

a[0].set_xlim([min(time), max(time)])

#fraud_df_time = transaction_data[transaction_data['Class']==1]



time_fraud = fraud_df['Time'].values



sns.distplot(time_fraud, ax = a[1])

a[1].set_title('Distribution with Time for Fraud Transactions')

#Keeping a similar scale as previous graph for comparison

a[1].set_xlim([min(time_fraud), max(time_fraud)])

plt.show()
#Scaling time and amount values, as all other predictors are already scaled through PCA

#Using min-max scaler for higher efficiency 



scaler = MinMaxScaler()

transaction_data['Time'] = scaler.fit_transform(transaction_data["Time"].values.reshape(-1,1))

transaction_data['Amount'] = scaler.fit_transform(transaction_data["Amount"].values.reshape(-1,1))
display(transaction_data.head())
#Splitting into values and class labels 

target = 'Class'

labels = transaction_data[target]

values = transaction_data.copy()

values = values.drop('Class', axis =1 )

display(values.head())

print("Dataset dimension :", values.shape)

print("Labels dimension :", labels.shape)
# We can use Stratified Shuffle Split or StratifiedKFold

strat_split = StratifiedShuffleSplit(n_splits=10, random_state=42)

for train_index, test_index in strat_split.split(values, labels):

    values_train, values_test = values.iloc[train_index], values.iloc[test_index]

    labels_train, labels_test = labels.iloc[train_index], labels.iloc[test_index]

print("Percentage of Fraud Transactions in Training Set using Stratified Shuffle Split:", 

      round(labels_train.value_counts()[1]/len(labels_train)*100, 4), "%")



kfold_split = StratifiedKFold(n_splits=10, random_state=47)

for train_index, test_index in kfold_split.split(values, labels):

    values_train1, values_test1 = values.iloc[train_index], values.iloc[test_index]

    labels_train1, labels_test1 = labels.iloc[train_index], labels.iloc[test_index]

print("Percentage of Fraud Transactions in Training Set using Stratified KFold:", 

      round(labels_train1.value_counts()[1]/len(labels_train1)*100, 4), "%")

#Applying SMOTE on the training dataset

smote = SMOTE(sampling_strategy='minority', random_state=47)

os_values, os_labels = smote.fit_sample(values_train, labels_train)



os_values = pd.DataFrame(os_values)

os_labels = pd.DataFrame(os_labels)



plt.figure(figsize=(5, 5))

sns.countplot(data = os_labels, x = 0 )

plt.title("Distribution of Oversampled Training Set")

plt.xlabel("Fraud")





print("Dimensions of Oversampled dataset is :", os_values.shape)
#SelectKBest on our oversampled training dataset

os_values_skb = os_values.copy()

skb = SelectKBest(k=15)

os_values_skb = skb.fit_transform(os_values_skb, os_labels[0].ravel())

display(pd.DataFrame(os_values_skb).head())
# Getting feature scores from SelectKBest after fitting

feature_list = values.columns

unsorted_list = zip(feature_list, skb.scores_)



sorted_features = sorted(unsorted_list, key=lambda x: x[1], reverse=True)

print(len(sorted_features))

print("Feature Scores:\n")

pprint(sorted_features[:15])
selected_features = [i[0] for i in sorted_features[:15]]

print(selected_features)

#Transforming test according to Feature Selection



#values_test = values_test[selected_features]

os_values_skb = os_values.copy()

display(values_test.head())
#Using this Oversampled data to fit a Logistic Regression model

logr = LogisticRegression().fit(os_values_skb, os_labels[0].ravel())



#Predicting on original test set

predictions = logr.predict(values_test)



print("Accuracy Score: ", round(accuracy_score(labels_test, predictions)*100, 2), '%')

print(classification_report(labels_test, predictions, target_names=["No Fraud", "Fraud"]))

#Computing major metrics

y_score = logr.decision_function(values_test)

precision, recall, _ = precision_recall_curve(labels_test, y_score)

fig = plt.figure(figsize=(10,5))

sns.lineplot(recall, precision, drawstyle='steps-post')

plt.fill_between(recall, precision, alpha=0.1, color='#78a2e2')

plt.title('Precision-Recall Curve for Logistic Regression')

plt.xlabel('Recall')

plt.ylabel('Precision')

plt.ylim([0.0, 1.1])

plt.xlim([0.0, 1.1])

t0 = time.time()



DT = DecisionTreeClassifier()

y_pred_DT = DT.fit(os_values_skb, os_labels[0].ravel()).predict(values_test)



t1 = time.time()

print("Fitting Decision Tree model took", t1-t0, "secs.")

print("."*10)

ada = AdaBoostClassifier()

y_pred_ada = ada.fit(os_values_skb, os_labels[0].ravel()).predict(values_test)



t2 = time.time()

print("Fitting Adaboost model took", t2-t1, "secs.")

print("."*10)

xgb = GradientBoostingClassifier()

y_pred_xgb = xgb.fit(os_values_skb, os_labels[0].ravel()).predict(values_test)



t3 = time.time()

print("Fitting XGBoost model took", t3-t2, "secs.")

print("."*10)
# Precision, Recall, and actual prediction numbers

print("Accuracy Score for DT: ", round(accuracy_score(labels_test, y_pred_DT)*100, 2), '%')

print(classification_report(labels_test, y_pred_DT))

print("-" * 40)

print("Accuracy Score for AdaBoost: ", round(accuracy_score(labels_test, y_pred_ada)*100, 2), '%')

print(classification_report(labels_test, y_pred_ada))

print("-" * 40)

print("Accuracy Score for XGB: ", round(accuracy_score(labels_test, y_pred_xgb)*100, 2), '%')

print(classification_report(labels_test, y_pred_xgb))
#Trying below three models:

# Logistic Regression Classifier

def tune_LogR():

    print("------------------ Using Logistic Regression --------------------")

    clf = LogisticRegression()

    param_grid = {

        'clf__penalty': ['l1', 'l2'],

        'clf__C' : [1.0, 10.0, 25.0, 50.0, 100.0, 500.0, 1000.0],

        'clf__solver' : ['liblinear']

    }



    return clf, param_grid
# Create pipeline

clf, params = tune_LogR()

estimators = [('clf', clf)]

pipe = Pipeline(estimators)



# Create GridSearchCV Instance

grid = GridSearchCV(pipe, params)

grid.fit(os_values_skb, os_labels[0].ravel())



# Final classifier

clf = grid.best_estimator_



print('\n=> Chosen parameters :')

print(grid.best_params_)



predictions = clf.predict(values_test)

print("Accuracy Score: ", round(accuracy_score(labels_test, predictions)*100, 2), '%')

print("Classification Report:\n", classification_report(labels_test, predictions, target_names = ['Non-Fraud', 'Fraud']))
#Plotting the P-R curve for this tuned model

y_score = clf.decision_function(values_test)

precision, recall, _ = precision_recall_curve(labels_test, y_score)

fig = plt.figure(figsize=(10,5))

sns.lineplot(recall, precision, drawstyle='steps-post')

plt.fill_between(recall, precision, alpha=0.1, color='#78a2e2')

plt.title('Precision-Recall Curve for Logistic Regression - After Tuning')

plt.xlabel('Recall')

plt.ylabel('Precision')

plt.ylim([0.0, 1.1])

plt.xlim([0.0, 1.1])
#Plotting confusion matrix for this

tn, fp, fn, tp = confusion_matrix(labels_test, predictions).ravel()

print("True Negatives:", tn)

print("False Positives:", fp)

print('False Negatives:', fn)

print('True Positives:', tp)
splitter = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=1)

for train_index, test_index in splitter.split(os_values, os_labels):

    _, os_values_test = os_values.iloc[train_index], os_values.iloc[test_index]

    _, os_labels_test = os_labels.iloc[train_index], os_labels.iloc[test_index]
predictions = clf.predict(os_values_test)

print("Accuracy Score: ", round(accuracy_score(os_labels_test, predictions)*100, 2), '%')

print("Classification Report:\n", classification_report(os_labels_test, predictions, target_names = ['Non-Fraud', 'Fraud']))