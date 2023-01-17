# Import all required libraries for reading, analysing and visualizing data

import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

import seaborn as sns
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# read the data from the csv file

records = pd.read_csv('/kaggle/input/indian-liver-patient-records/indian_liver_patient.csv')
display(records.head())
records.shape
records.info()
# describe gives statistical information about NUMERICAL columns in the dataset

records.describe(include = 'all')
# check if any of the columns has null values

records.isnull().sum()
# replace 2 in 'Dataset' column with 0

records['Dataset'] = records['Dataset'].replace(2, 0)
# make the gender column into numerical format

records['Gender'] = records['Gender'].apply(lambda x: 1 if x == 'Male' else 0)
# 'Albumin_and_Globulin_Ratio' column contains 4 null values, so fill these values with the mean of the column values

records[records['Albumin_and_Globulin_Ratio'].isnull()]

records["Albumin_and_Globulin_Ratio"] = records.Albumin_and_Globulin_Ratio.fillna(records['Albumin_and_Globulin_Ratio'].mean())
# visualize number of patients diagonised with liver diesease

sns.countplot(data = records, x = 'Dataset');



p1, p2 = records['Dataset'].value_counts()

print('Number of people diagonised with liver disease: ', p1)

print('Number of people not diagonised with liver disease: ', p2)
sns.catplot(data = records, x = 'Gender', y = 'Age', hue = 'Dataset', jitter = 0.4);
sns.catplot(x='Dataset', y='Total_Bilirubin', data=records, kind = 'boxen', color = 'green');
sns.catplot(x='Dataset', y='Direct_Bilirubin', data=records, kind = 'boxen', color = 'blue');
sns.catplot(x='Dataset', y='Alkaline_Phosphotase', data=records, kind = 'boxen', color = 'red');
sns.catplot(x='Dataset', y='Alamine_Aminotransferase', data=records, kind = 'boxen', color = 'orange');
sns.catplot(x='Dataset', y='Aspartate_Aminotransferase', data=records, kind = 'boxen', color = 'brown');
sns.catplot(x='Dataset', y='Total_Protiens', data=records, kind = 'boxen', color = 'purple');
sns.catplot(x='Dataset', y='Albumin', data=records, kind = 'boxen', color = 'violet');
sns.catplot(x='Dataset', y='Albumin_and_Globulin_Ratio', data=records, kind = 'boxen', color = 'black');
plt.figure(figsize=(12, 10))

plt.title('Correlation between features');

sns.heatmap(records.corr(), annot = True, fmt = '0.2f');
# import the required modules

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn import svm

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
data = records.values

X = data[:, :-1]

Y = data[:, -1]
scaler = MinMaxScaler()

X = scaler.fit_transform(X)
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.3, random_state = 123)

Y_train = Y_train.reshape(408, 1)

Y_test = Y_test.reshape(175, 1)

print("X_train shape:" + str(X_train.shape))

print("Y_train shape:" + str(Y_train.shape))

print("X_test shape:" + str(X_test.shape))

print("Y_test shape:" + str(Y_test.shape))
logreg = LogisticRegression()

logreg.fit(X_train, Y_train)



# train and test scores

lr_train_score = round(logreg.score(X_train, Y_train) * 100, 2)

lr_test_score = round(logreg.score(X_test, Y_test) * 100, 2)

# predicted output

Y_pred_lr = logreg.predict(X_test)



print('Logistic Regression train score: ', lr_train_score)

print('Logistic Regression test score: ', lr_test_score)

print('Classification Report: \n', classification_report(Y_test, Y_pred_lr))

print('Confusion Matrix:\n', confusion_matrix(Y_test, Y_pred_lr))
rf_cl = RandomForestClassifier()

rf_cl.fit(X_train, Y_train)



# train and test scores

rf_train_score = round(rf_cl.score(X_train, Y_train) * 100, 2)

rf_test_score = round(rf_cl.score(X_test, Y_test) * 100, 2)

# predicted output

Y_pred_rf = rf_cl.predict(X_test)



print('Random Forest train score: ', rf_train_score)

print('Random Forest test score: ', rf_test_score)

print('Classification Report: \n', classification_report(Y_test, Y_pred_rf))

print('Confusion Matrix:\n', confusion_matrix(Y_test, Y_pred_rf))
svm_cl = svm.SVC()

svm_cl.fit(X_train, Y_train)



# train and test scores

svm_train_score = round(svm_cl.score(X_train, Y_train) * 100, 2)

svm_test_score = round(svm_cl.score(X_test, Y_test) * 100, 2)

# predicted output

Y_pred_svm = svm_cl.predict(X_test)



print('SVM train score: ', svm_train_score)

print('SVM score: ', svm_test_score)

print('Classification Report: \n', classification_report(Y_test, Y_pred_svm))

print('Confusion Matrix:\n', confusion_matrix(Y_test, Y_pred_svm))