from sklearn import decomposition

import pandas as pd

import numpy as np

from sklearn import preprocessing

import matplotlib.pyplot as plt 

plt.rc("font", size=14)

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

import scikitplot as skplt



import seaborn as sns

sns.set(style="white")

sns.set(style="whitegrid", color_codes=True)





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_raw = pd.read_excel('/kaggle/input/covid19/dataset.xlsx')

df_raw.head(5)
df_raw.describe()
df_raw.isnull().sum().head(100)
# Define the dataset



df = df_raw.fillna(-99).dropna()

new_columns=[]

for i in df.columns:

    if i != 'Patient ID':

        if i != 'SARS-Cov-2 exam result':

            if i != 'Patient addmited to regular ward (1=yes, 0=no)':

                if i != 'Patient addmited to semi-intensive unit (1=yes, 0=no)':

                    if i != 'Patient addmited to intensive care unit (1=yes, 0=no)':

                        new_columns.append(i)

        

X0 = df[new_columns]

# Define features

X1 = pd.get_dummies(X0)  # Creating dummy variables  

X1.head(5)
# Check for balance

FIGURE_SIZE = (8,8)

df['SARS-Cov-2 exam result'].value_counts().plot.pie(figsize=FIGURE_SIZE)
# Define encoded target 

y = df[['SARS-Cov-2 exam result']]

y = y.replace({'negative': 0, 'positive': 1})

y
# Run the Recursive feature elimination

logregHasCovid = LogisticRegression()

n_features_to_select = 45

rfe = RFE(logregHasCovid, n_features_to_select)

rfe = rfe.fit(X1, y.values.ravel())

# print(rfe.support_)

# print(rfe.ranking_)
# Define new columns from RFE

columns_retained_RFE_HasCovid = X1.columns[rfe.get_support()].values

columns_retained_RFE_HasCovid
# Define new training features

X = X1[columns_retained_RFE_HasCovid]

X.head(4)
# Train logistic regression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=48)



logregHasCovid = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,

                   intercept_scaling=1, l1_ratio=None, max_iter=500,

                   multi_class='auto', n_jobs=None, penalty='l2',

                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,

                   warm_start=False)



logregHasCovid.fit(X_train, y_train)

y_pred = logregHasCovid.predict(X_test)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logregHasCovid.score(X_test, y_test)))
# Results with confusion matrix

FIGURE_SIZE = (8,8)

confusion_matrix0 = confusion_matrix(y_test, y_pred)

skplt.metrics.plot_confusion_matrix(y_test, y_pred, figsize=FIGURE_SIZE)
# Results with classification_report



from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
# Results with ROC

logit_roc_auc = roc_auc_score(y_test, logregHasCovid.predict(X_test))

fpr, tpr, thresholds = roc_curve(y_test, logregHasCovid.predict_proba(X_test)[:,1])

plt.figure(figsize=FIGURE_SIZE)

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.savefig('Log_ROC')

plt.show()
# del X, X0, X1, y
df = df_raw.fillna(-99).dropna()

new_columns2=[]

for i in df.columns:

    if i != 'Patient ID':

        if i != 'Patient addmited to regular ward (1=yes, 0=no)':

            if i != 'Patient addmited to semi-intensive unit (1=yes, 0=no)':

                if i != 'Patient addmited to intensive care unit (1=yes, 0=no)':

                    new_columns2.append(i)

        

X = df[new_columns2]

X = pd.get_dummies(X)  # Creating dummy variables  



X.head(5)

X.columns
# Check for balance

df['Patient addmited to regular ward (1=yes, 0=no)'].value_counts().plot.pie(figsize=FIGURE_SIZE)
y = df[['Patient addmited to regular ward (1=yes, 0=no)']]
logregRegular = LogisticRegression()

n_features_to_select = 24

rfe = RFE(logregRegular, n_features_to_select)

rfe = rfe.fit(X, y.values.ravel())

# print(rfe.support_)

# print(rfe.ranking_)
columns_retained_RFE_Regular = X.columns[rfe.get_support()].values

columns_retained_RFE_Regular
X = X[columns_retained_RFE_Regular]

X.head(4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)



logregRegular = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,

                   intercept_scaling=1, l1_ratio=None, max_iter=500,

                   multi_class='auto', n_jobs=None, penalty='l2',

                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,

                   warm_start=False)



logregRegular.fit(X_train, y_train)

y_pred = logregRegular.predict(X_test)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logregRegular.score(X_test, y_test)))
confusion_matrix1 = confusion_matrix(y_test, y_pred)

skplt.metrics.plot_confusion_matrix(y_test, y_pred, figsize=FIGURE_SIZE)

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
logit_roc_auc = roc_auc_score(y_test, logregRegular.predict(X_test))

fpr, tpr, thresholds = roc_curve(y_test, logregRegular.predict_proba(X_test)[:,1])

plt.figure(figsize=FIGURE_SIZE)

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.savefig('Log_ROC')

plt.show()
# Check for balance

df['Patient addmited to semi-intensive unit (1=yes, 0=no)'].value_counts().plot.pie(figsize=FIGURE_SIZE)
# Reset variables

y = df[['Patient addmited to semi-intensive unit (1=yes, 0=no)']]

X = df[new_columns2]

X = pd.get_dummies(X)  # Creating dummy variables  



# Recursive Feature Elimination (RFE)

logregSemiIntensive = LogisticRegression()

n_features_to_select = 24

rfe = RFE(logregSemiIntensive, n_features_to_select, verbose=False)

rfe = rfe.fit(X, y.values.ravel())

# print(rfe.support_)

# print(rfe.ranking_)




# Replace the dataset with new features

columns_retained_RFE_SemiIntensive = X.columns[rfe.get_support()].values



X = X[columns_retained_RFE_SemiIntensive]





# Split train and test dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)



# Train model

logregSemiIntensive = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,

                   intercept_scaling=1, l1_ratio=None, max_iter=500,

                   multi_class='auto', n_jobs=None, penalty='l2',

                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,

                   warm_start=False)





logregSemiIntensive.fit(X_train, y_train)

y_pred = logregSemiIntensive.predict(X_test)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logregSemiIntensive.score(X_test, y_test)))



confusion_matrix2 = confusion_matrix(y_test, y_pred)

skplt.metrics.plot_confusion_matrix(y_test, y_pred, figsize=FIGURE_SIZE)



print(classification_report(y_test, y_pred))



logit_roc_auc = roc_auc_score(y_test, logregSemiIntensive.predict(X_test))

fpr, tpr, thresholds = roc_curve(y_test, logregSemiIntensive.predict_proba(X_test)[:,1])

plt.figure(figsize=FIGURE_SIZE)

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.savefig('Log_ROC')

plt.show()
# Check for balance

df['Patient addmited to intensive care unit (1=yes, 0=no)'].value_counts().plot.pie(figsize=FIGURE_SIZE)
y = df[['Patient addmited to intensive care unit (1=yes, 0=no)']]

X = df[new_columns2]

X = pd.get_dummies(X)  # Creating dummy variables  



# Recursive Feature Elimination (RFE)

logregIntensive = LogisticRegression()

n_features_to_select = 40

rfe = RFE(logregIntensive, n_features_to_select)

rfe = rfe.fit(X, y.values.ravel())

# print(rfe.support_)

# print(rfe.ranking_)




# Replace the dataset with new features

columns_retained_RFE_Intensive = X.columns[rfe.get_support()].values



X = X[columns_retained_RFE_Intensive]





# Split train and test dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)



# Train model

logregIntensive = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,

                   intercept_scaling=1, l1_ratio=None, max_iter=500,

                   multi_class='auto', n_jobs=None, penalty='l2',

                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,

                   warm_start=False)





logregIntensive.fit(X_train, y_train)

y_pred = logregIntensive.predict(X_test)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logregIntensive.score(X_test, y_test)))

confusion_matrix3 = confusion_matrix(y_test, y_pred)

skplt.metrics.plot_confusion_matrix(y_test, y_pred, figsize=FIGURE_SIZE)



print(classification_report(y_test, y_pred))





logit_roc_auc = roc_auc_score(y_test, logregIntensive.predict(X_test))

fpr, tpr, thresholds = roc_curve(y_test, logregIntensive.predict_proba(X_test)[:,1])

plt.figure(figsize=FIGURE_SIZE)

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.savefig('Log_ROC')

plt.show()
# Function for evaluating

def evaluate_sample(sample_number):

    X = df[new_columns2]

    X = pd.get_dummies(X)  # Creating dummy variables  



    X1 = X[columns_retained_RFE_Regular].values[sample_number].reshape(1, -1)

    X2 = X[columns_retained_RFE_SemiIntensive].values[sample_number].reshape(1, -1)

    X3 = X[columns_retained_RFE_Intensive].values[sample_number].reshape(1, -1)



    print(f'Probabilities for regular treatment. No: {logregRegular.predict_proba(X1)[0][0]:.4f},  Yes: {logregRegular.predict_proba(X1)[0][1]:.4f}')

    print(f'Probabilities for semi intensive treatment. No: {logregSemiIntensive.predict_proba(X2)[0][0]:.4f},  Yes: {logregSemiIntensive.predict_proba(X2)[0][1]:.4f}')

    print(f'Probabilities for intensive treatment. No: {logregIntensive.predict_proba(X3)[0][0]:.4f},  Yes: {logregIntensive.predict_proba(X3)[0][1]:.4f}')

    

    print('\n.............Conclusion: ........................\n')



    if logregIntensive.predict_proba(X3)[0][1] < 0.51 and logregIntensive.predict_proba(X3)[0][1] > 0.1:

        print('Patient with moderate probability of receiving intensive treatment.')     

    elif logregIntensive.predict_proba(X3)[0][1] > 0.51:

        print('Patient with high probability of receiving intensive treatment.')   

        

    elif logregSemiIntensive.predict_proba(X2)[0][1] < 0.51 and logregSemiIntensive.predict_proba(X2)[0][1] > 0.1:

        print('Patient with moderate probability of receiving semi intensive treatment.')        

    elif logregSemiIntensive.predict_proba(X2)[0][1] > 0.51:

        print('Patient with high probability of receiving semi intensive treatment.') 

        

    elif logregRegular.predict_proba(X1)[0][1] > 0.1 and logregRegular.predict_proba(X1)[0][1] < 0.50:

        print('Patient with moderate probability of receiving treatment in the regular ward.') 

    elif logregRegular.predict_proba(X1)[0][1] > 0.51:

        print('Patient with high probability of receiving treatment in the regular ward.')    

        

    else:

        print('Patient without the need for hospitalization.')
evaluate_sample(5)
evaluate_sample(5226)
evaluate_sample(5009)
evaluate_sample(111)