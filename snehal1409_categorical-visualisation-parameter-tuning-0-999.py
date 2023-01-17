# Load Libraries



import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



# Import libraries for data transformation

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split





# Import classification

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier





from sklearn import metrics

from sklearn.feature_selection import RFE

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import roc_curve, auc







import time

import warnings

warnings.filterwarnings("ignore")

print(os.listdir("../input"))

df  = pd.read_csv('../input/mushrooms.csv')

df.head()
df.describe()
df['class'].value_counts(normalize = True) # Check Class Balance
analysis = [df.shape, df.columns, df.info(), df.isnull().sum()]

for j in analysis:

    print(j,'\n----------------------------------------------------------------------------\n')
fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(25, 25),sharey=True)

fig.subplots_adjust(hspace=1.2, wspace=0.6)





for ax, col in zip(axes[0], df.iloc[:,1:].columns):

    pd.crosstab(df[col], df['class']).plot.bar(stacked=True, ax = ax)

    plt.xlabel(col, fontsize=18)

    #ax.set_title(col)

for ax, col in zip(axes[1], df.iloc[:,5:].columns):

    pd.crosstab(df[col], df['class']).plot.bar(stacked=True, ax = ax)

    plt.xlabel(col, fontsize=18)

    #ax.set_title(col)

for ax, col in zip(axes[2], df.iloc[:,9:].columns):

    pd.crosstab(df[col], df['class']).plot.bar(stacked=True, ax = ax)

    plt.xlabel(col, fontsize=18)

    #ax.set_title(col)

for ax, col in zip(axes[3], df.iloc[:,13:].columns):

    pd.crosstab(df[col], df['class']).plot.bar(stacked=True, ax = ax)

    plt.xlabel(col, fontsize=18)

    #ax.set_title(col)

for ax, col in zip(axes[4], df.iloc[:,17:].columns):

    pd.crosstab(df[col], df['class']).plot.bar(stacked=True, ax = ax)

    plt.xlabel(col, fontsize=18)

    #ax.set_title(col)





#handles, labels = ax.get_legend_handles_labels()

#fig.legend(handles, labels, loc='upper center')

plt.tight_layout()

df.groupby(['gill-attachment','ring-number','veil-color'])['class'].value_counts(normalize=True).unstack()
data = df.drop(['gill-attachment','ring-number','veil-color','veil-type'], axis=1)

data.shape
y = data.iloc[:,0]

data_X = data.drop('class', axis=1)
# Label Encoding

le=LabelEncoder()

y=le.fit_transform(y)



#One-Hot encoding

X = pd.DataFrame()

for variables in data_X:

    dummy_var = pd.get_dummies(data_X[variables],prefix=variables)

    X = pd.concat([X,dummy_var], axis=1)



X.head()
X.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=101)
lr = LogisticRegression()

lr.get_params().keys()
start = time.clock()



parameters = {'solver':('liblinear', 'newton-cg', 'lbfgs', 'sag'), 

              'C': np.logspace(1,0.1,10),

             'max_iter': [200],

             'n_jobs':[-1]}



clf = GridSearchCV(lr, parameters, cv=5, verbose=1)

best_model = clf.fit(X, y)

  

print('Time taken (in secs) to tune hyperparameters: {:.2f}'.format(time.clock() - start))
y_pred = clf.predict(X_test)

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)

roc_auc = auc(false_positive_rate, true_positive_rate)



print(clf.best_estimator_)

print('\n---------------------------------------------------------------------------------')

print('Confusion matrix of logistic regression classifier on test set: \n{}'.format(metrics.confusion_matrix(y_test,y_pred)))

print('\n---------------------------------------------------------------------------------')

print('Accuracy of logistic regression classifier on test set: {:.10f}'.format(clf.score(X_test, y_test)))

print('\n---------------------------------------------------------------------------------')

print('ROC of logistic regression classifier on test set: {:.10f}'.format(roc_auc))

plt.figure()

lw = 2

plt.plot(false_positive_rate[2], true_positive_rate[2], color='darkorange',

         lw=lw, label='ROC curve (area = %0.2f)'% roc_auc)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic example')

plt.legend(loc="lower right")

plt.show()