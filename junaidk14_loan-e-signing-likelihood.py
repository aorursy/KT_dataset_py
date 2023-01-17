#### Importing Libraries ####



import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sn
dataset = pd.read_csv('../input/financial_data.csv')
### EDA ###



dataset.head()
dataset.columns
dataset.describe().T
## Cleaning Data



# Removing NaN

dataset.isna().any() # No NAs
## Histograms



dataset2 = dataset.drop(columns = ['entry_id', 'pay_schedule', 'e_signed'])



fig = plt.figure(figsize=(15, 12))

plt.suptitle('Histograms of Numerical Columns', fontsize=20)

for i in range(dataset2.shape[1]):

    plt.subplot(6, 3, i + 1)

    f = plt.gca()

    f.set_title(dataset2.columns.values[i])



    vals = np.size(dataset2.iloc[:, i].unique())

    if vals >= 100:

        vals = 100

    

    plt.hist(dataset2.iloc[:, i], bins=vals, color='#3F5D7D')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
## Correlation with Response Variable (Note: Models like RF are not linear like these)



dataset2.corrwith(dataset.e_signed).plot.bar(figsize = (20, 10), title = "Correlation with E Signed", fontsize = 15, rot = 45, grid = True)
## Correlation Matrix

sn.set(style="white")
# Compute the correlation matrix

corr = dataset2.corr()
# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True
# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(18, 15))



# Generate a custom diverging colormap

cmap = sn.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sn.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})

import random

import time
random.seed(100)
# Feature Engineering



dataset = dataset.drop(columns = ['months_employed'])

dataset['personal_account_months'] = (dataset.personal_account_m + (dataset.personal_account_y * 12))

dataset[['personal_account_m', 'personal_account_y', 'personal_account_months']].head()

dataset = dataset.drop(columns = ['personal_account_m', 'personal_account_y'])
# One Hot Encoding

dataset = pd.get_dummies(dataset)

dataset.columns

dataset = dataset.drop(columns = ['pay_schedule_semi-monthly'])



# Removing extra columns

response = dataset["e_signed"]

users = dataset['entry_id']

dataset = dataset.drop(columns = ["e_signed", "entry_id"])
# Splitting into Train and Test Set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(dataset,response,test_size = 0.2,random_state = 0)
# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train2 = pd.DataFrame(sc_X.fit_transform(X_train))

X_test2 = pd.DataFrame(sc_X.transform(X_test))

X_train2.columns = X_train.columns.values

X_test2.columns = X_test.columns.values

X_train2.index = X_train.index.values

X_test2.index = X_test.index.values

X_train = X_train2

X_test = X_test2
###Comparing Models



## Logistic Regression

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0, penalty = 'l1')

classifier.fit(X_train, y_train)



# Predicting Test Set

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

acc = accuracy_score(y_test, y_pred)

prec = precision_score(y_test, y_pred)

rec = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



results = pd.DataFrame([['Linear Regression (Lasso)', acc, prec, rec, f1]], columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
## SVM (Linear)

from sklearn.svm import SVC

classifier = SVC(random_state = 0, kernel = 'linear')

classifier.fit(X_train, y_train)



# Predicting Test Set

y_pred = classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)

prec = precision_score(y_test, y_pred)

rec = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



model_results = pd.DataFrame([['SVM (Linear)', acc, prec, rec, f1]], columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])



results = results.append(model_results, ignore_index = True)
## SVM (rbf)

from sklearn.svm import SVC

classifier = SVC(random_state = 0, kernel = 'rbf')

classifier.fit(X_train, y_train)



# Predicting Test Set

y_pred = classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)

prec = precision_score(y_test, y_pred)

rec = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



model_results = pd.DataFrame([['SVM (RBF)', acc, prec, rec, f1]], columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])



results = results.append(model_results, ignore_index = True)
##  Random Forest

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(random_state = 0, n_estimators = 100, criterion = 'entropy')

classifier.fit(X_train, y_train)



# Predicting Test Set

y_pred = classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)

prec = precision_score(y_test, y_pred)

rec = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



model_results = pd.DataFrame([['Random Forest (n=100)', acc, prec, rec, f1]], columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])



results = results.append(model_results, ignore_index = True)
# Formatting Final Results



final_results = pd.concat([y_test, users], axis = 1).dropna()

final_results['predictions'] = y_pred

final_results = final_results[['entry_id', 'e_signed', 'predictions']]
final_results