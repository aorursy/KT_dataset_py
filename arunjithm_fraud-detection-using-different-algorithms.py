# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LogisticRegressionCV

from sklearn.model_selection import GridSearchCV

from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import MinMaxScaler

from sklearn.naive_bayes import GaussianNB
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.metrics import precision_score,confusion_matrix, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import KFold, cross_val_score
%matplotlib inline
df = pd.read_csv("../input/creditcardfraud/creditcard.csv")
df.describe()
print('The total number of transactions in dataset : ', len(df))

print('The total number of columns : ',len(list(df)))

print('The dimension of data : ', df.shape)

print('The target column is : ', list(df)[30])

print('Total number of unique values in target column is : ', len(df['Class'].unique()))

print('The unique values in Class column : ', df.Class.unique())
print('Total number of zeroes (non-fraud transactions) : ', df['Class'].value_counts()[0])

print('Total number of ones (fraud transactions) : ', df['Class'].value_counts()[1])

print('Percentage of non-fraud transactions : ', 100*(df['Class'].value_counts()[0])/ len(df))

print('Percentage of fraud transactions : ', 100*(df['Class'].value_counts()[1])/ len(df))

sns.countplot('Class', data=df, palette=None)

plt.title("Target Column frequency distribution")

plt.xlabel("Class")

plt.ylabel("Frequency")
df.plot(x='Time', y='Amount', style='-')

plt.title("Transaction Amount vs Time")

plt.xlabel("Time")

plt.ylabel("Amount")
df = df.drop(['Time'],axis=1)
df['Normalized_Amount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))

df = df.drop(['Amount'],axis=1)

df.head()
Normalized_Amount = df['Normalized_Amount']

df=df.drop(['Normalized_Amount'],axis=1)

df.insert(0, 'Normalized_Amount', Normalized_Amount)

df.head()
flag = df.isnull().sum().any()



if (flag == True):

    df.isnull().sum()

    print("There are null values in the dataframe")

    

else :

    print("There are no null values and dataframe is clear for further analysis")
# ignore all future warnings

from warnings import simplefilter



simplefilter(action='ignore', category=FutureWarning)
X = df.drop(['Class'], axis = 1) 

y = df['Class']



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 2)

clf = LogisticRegression().fit(X_train, y_train)

print('Accuracy of Logistic regression classifier on imbalanced training set: {:.2f}'.format(clf.score(X_train, y_train)))

print('Accuracy of Logistic regression classifier on imbalanced test set: {:.2f}'.format(clf.score(X_test, y_test)))

lr = LogisticRegression().fit(X_train, y_train)

lr_predicted = lr.predict(X_test)

confusion = confusion_matrix(y_test, lr_predicted)



print('Logistic regression classifier (default settings)\n', confusion)
print("Logistic Regression Evaluation Parameters with imbalanced data")

print('Accuracy: {:.2f}'.format(accuracy_score(y_test, lr_predicted)))

print('Precision: {:.2f}'.format(precision_score(y_test, lr_predicted)))

print('Recall: {:.2f}'.format(recall_score(y_test, lr_predicted)))

print('F1: {:.2f}'.format(f1_score(y_test, lr_predicted)))
scaler = MinMaxScaler()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)



clf = MLPClassifier(hidden_layer_sizes = [100, 100], alpha = 5.0,

                   random_state = 0, solver='lbfgs').fit(X_train_scaled, y_train)



print('Accuracy of NN classifier on training set: {:.2f}'.format(clf.score(X_train_scaled, y_train)))

print('Accuracy of NN classifier on test set: {:.2f}'.format(clf.score(X_test_scaled, y_test)))
nbclf = GaussianNB().fit(X_train, y_train)

print('Accuracy of GaussianNB classifier on training set: {:.2f}'.format(nbclf.score(X_train, y_train)))

print('Accuracy of GaussianNB classifier on test set: {:.2f}'.format(nbclf.score(X_test, y_test)))
#Recalling the amount of unique values in 'Class' column

print('Total number of zeroes (non-fraud transactions) : ', df['Class'].value_counts()[0])

print('Total number of ones (fraud transactions) : ', df['Class'].value_counts()[1])
non_fraud_transactions_df = df[df['Class'] == 0]

fraud_transactions_df = df[df['Class']==1]
print('The dimension of fraud transactions dataframe is : ', fraud_transactions_df.shape)

print('The dimension of non-fraud transactions dataframe is : ', non_fraud_transactions_df.shape)
sample_492_non_fraud_transactions_df = non_fraud_transactions_df.sample(n=492)

print('The dimension of sample non-fraud transactions df is : ', sample_492_non_fraud_transactions_df.shape)
method_1_df = pd.concat([sample_492_non_fraud_transactions_df, fraud_transactions_df])

method_1_df = method_1_df.sample(frac=1).reset_index(drop=True)

method_1_df.head()
print('The dimension of dataframe for Method 1 is : ',method_1_df.shape )
sns.countplot('Class', data=method_1_df, palette=None)

plt.title("Method 1 Frequency distribution plot")

plt.xlabel("Class")

plt.ylabel("Frequency")
print('Percentage of non-fraud transactions in method_1_df : ',  100*(method_1_df['Class'].value_counts()[0])/ len(method_1_df))

print('Percentage of fraud transactions in method_1_df : ',  100*(method_1_df['Class'].value_counts()[1])/ len(method_1_df))
X = method_1_df.drop(['Class'], axis = 1) 

y = method_1_df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=27)
print("Number transactions in training dataset for Method 1: ", len(X_train))

print("Number transactions in testing dataset  for Method 1: ", len(X_test))

print("Total number of transactions  for Method 1 : ", len(X_train)+len(X_test))
clf = LogisticRegression().fit(X_train, y_train)

print('Accuracy of Logistic regression classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))

print('Accuracy of Logistic regression classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))
lr_predicted = clf.predict(X_test)

confusion = confusion_matrix(y_test, lr_predicted)

print('Logistic regression classifier (default settings)\n', confusion)
print('Accuracy: {:.2f}'.format(accuracy_score(y_test, lr_predicted)))

print('Precision: {:.2f}'.format(precision_score(y_test, lr_predicted)))

print('Recall: {:.2f}'.format(recall_score(y_test, lr_predicted)))

print('F1: {:.2f}'.format(f1_score(y_test, lr_predicted)))
logistic_parameters = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

grid = GridSearchCV(LogisticRegression(), logistic_parameters)

grid.fit(X_train, y_train)

best_log_reg = grid.best_estimator_
logistic_score = cross_val_score(best_log_reg, X_train, y_train, cv=5)

print('Logistic Regression Cross Validation Score: ', logistic_score.mean())
lr_predicted = grid.predict(X_test)

confusion = confusion_matrix(y_test, lr_predicted)

print('Logistic regression classifier with Cross-validation (default settings)\n', confusion)
print('Accuracy: {:.2f}'.format(accuracy_score(y_test, lr_predicted)))

print('Precision: {:.2f}'.format(precision_score(y_test, lr_predicted)))

print('Recall: {:.2f}'.format(recall_score(y_test, lr_predicted)))

print('F1: {:.2f}'.format(f1_score(y_test, lr_predicted)))
nbclf = GaussianNB().fit(X_train, y_train)

print('Accuracy of GaussianNB classifier on training set: {:.2f}'.format(nbclf.score(X_train, y_train)))

print('Accuracy of GaussianNB classifier on test set: {:.2f}'.format(nbclf.score(X_test, y_test)))
X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

clf = MLPClassifier(hidden_layer_sizes = [100, 100], alpha = 5.0,

                   random_state = 0, solver='lbfgs').fit(X_train_scaled, y_train)



print('Accuracy of NN classifier on training set: {:.2f}'.format(clf.score(X_train_scaled, y_train)))

print('Accuracy of NN classifier on test set: {:.2f}'.format(clf.score(X_test_scaled, y_test)))
print('The dimension of fraud transactions dataframe is : ', fraud_transactions_df.shape)

print('The dimension of non-fraud transactions dataframe is : ', non_fraud_transactions_df.shape)
len(non_fraud_transactions_df) / len(fraud_transactions_df)
upsampled_df = pd.concat([fraud_transactions_df] * 577, ignore_index=True)
print('The dimension of upsampled fraud transactions dataframe is : ', upsampled_df.shape)

print('The dimension of non-fraud transactions dataframe is : ', non_fraud_transactions_df.shape)
print('Difference in number of rows between two dataframes after upsampling is : ', len(non_fraud_transactions_df) - len(upsampled_df))
upsampled_df.describe()
method_2_df = pd.concat([upsampled_df, non_fraud_transactions_df])

method_2_df = method_2_df.sample(frac=1).reset_index(drop=True)

method_2_df.describe()
print('The dimension of method_2_df is :', method_2_df.shape)
sns.countplot('Class', data=method_2_df, palette=None)

plt.title("Method 2 Frequency distribution plot")

plt.xlabel("Class")

plt.ylabel("Frequency")
print('Percentage of non-fraud transactions in method_2_df : ',  100*(method_2_df['Class'].value_counts()[0])/ len(method_2_df))

print('Percentage of fraud transactions in method_2_df : ',  100*(method_2_df['Class'].value_counts()[1])/ len(method_2_df))
X = method_2_df.drop(['Class'], axis = 1) 

y = method_2_df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=27)
print("Number transactions in training dataset for Method 2: ", len(X_train))

print("Number transactions in testing dataset  for Method 2: ", len(X_test))

print("Total number of transactions  for Method 2 : ", len(X_train)+len(X_test))
clf = LogisticRegression().fit(X_train, y_train)

print('Accuracy of Logistic regression classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))

print('Accuracy of Logistic regression classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))
lr_predicted = clf.predict(X_test)

confusion = confusion_matrix(y_test, lr_predicted)

print('Logistic regression classifier (default settings)\n', confusion)
print('Accuracy: {:.2f}'.format(accuracy_score(y_test, lr_predicted)))

print('Precision: {:.2f}'.format(precision_score(y_test, lr_predicted)))

print('Recall: {:.2f}'.format(recall_score(y_test, lr_predicted)))

print('F1: {:.2f}'.format(f1_score(y_test, lr_predicted)))
logistic_parameters = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

grid = GridSearchCV(LogisticRegression(), logistic_parameters)

grid.fit(X_train, y_train)

best_log_reg = grid.best_estimator_

logistic_score = cross_val_score(best_log_reg, X_train, y_train, cv=5)

print('Logistic Regression Cross Validation Score: ', logistic_score.mean())
lr_predicted = grid.predict(X_test)

confusion = confusion_matrix(y_test, lr_predicted)

print('Logistic regression classifier with Cross-validation (default settings)\n', confusion)
print('Accuracy: {:.2f}'.format(accuracy_score(y_test, lr_predicted)))

print('Precision: {:.2f}'.format(precision_score(y_test, lr_predicted)))

print('Recall: {:.2f}'.format(recall_score(y_test, lr_predicted)))

print('F1: {:.2f}'.format(f1_score(y_test, lr_predicted)))
nbclf = GaussianNB().fit(X_train, y_train)

print('Accuracy of GaussianNB classifier on training set: {:.2f}'.format(nbclf.score(X_train, y_train)))

print('Accuracy of GaussianNB classifier on test set: {:.2f}'.format(nbclf.score(X_test, y_test)))
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)



clf = MLPClassifier(hidden_layer_sizes = [100, 100], alpha = 5.0,

                   random_state = 0, solver='lbfgs').fit(X_train_scaled, y_train)



print('Accuracy of NN classifier on training set: {:.2f}'.format(clf.score(X_train_scaled, y_train)))

print('Accuracy of NN classifier on test set: {:.2f}'.format(clf.score(X_test_scaled, y_test)))