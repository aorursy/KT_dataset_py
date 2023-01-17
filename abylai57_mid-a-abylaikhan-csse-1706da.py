# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#importing all the libraries needed

import seaborn as sns

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

from sklearn import metrics

import random

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from sklearn.model_selection import train_test_split

%matplotlib inline

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

from sklearn.metrics import mean_squared_error

#--------------------------------------------------

#SVM

import numpy as np

#import cvxopt

from sklearn.datasets.samples_generator import make_blobs

from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

from sklearn.svm import LinearSVC

from sklearn.metrics import confusion_matrix

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

%matplotlib inline

from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report,accuracy_score,confusion_matrix

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data_train = pd.read_csv("/kaggle/input/santander-customer-transaction-prediction/train.csv")

data_test = pd.read_csv("/kaggle/input/santander-customer-transaction-prediction/test.csv")

data_submission = pd.read_csv("/kaggle/input/santander-customer-transaction-prediction/sample_submission.csv")
df_t = data_train

df_test = data_test
df_test.shape
df_t.shape
print ('Train shape : ',df_t.shape)

print ('Test shape : ',df_test.shape)
df_t.head(10)
df_t.isnull().sum()
df_t.info()
df_t.describe()
df_t.target.unique()
df_t.target.value_counts()
df_t.corr()
df_t.iloc[2:5, 2:10].corr()
plt.figure(figsize=(10, 7))

sns.heatmap(df_t.corr())
plt.figure(figsize=(10,10))

sns.heatmap(df_t.iloc[2:5, 2:10].corr(), annot = True, vmin=-1, vmax=1, center= 0)
def plot_feature_scatter(df1, df2, features):

    i = 0

    sns.set_style('whitegrid')

    plt.figure()

    fig, ax = plt.subplots(4,4,figsize=(14,14))



    for feature in features:

        i += 1

        plt.subplot(4,4,i)

        plt.scatter(df1[feature], df2[feature], marker='+')

        plt.xlabel(feature, fontsize=9)

    plt.show();
features = ['var_0', 'var_1','var_2','var_3', 'var_4', 'var_5', 'var_6', 'var_7', 

           'var_8', 'var_9', 'var_10','var_11','var_12', 'var_13', 'var_14', 'var_15', 

           ]

plot_feature_scatter(df_t[::20],df_test[::20], features)
sns.countplot(df_t['target'], palette='Set3')
plt.figure(figsize=(10,6))

features = df_t.columns.values[2:202]

plt.title("Distribution of mean values per row in the train and test set")

sns.distplot(df_t[features].mean(axis=1),color="green", kde=True,bins=120, label='train')

sns.distplot(df_test[features].mean(axis=1),color="blue", kde=True,bins=120, label='test')

plt.legend()

plt.show()
plt.figure(figsize=(10,6))

plt.title("Distribution of mean values per column in the train and test set")

sns.distplot(df_t[features].mean(axis=0),color="magenta",kde=True,bins=120, label='train')

sns.distplot(df_test[features].mean(axis=0),color="darkblue", kde=True,bins=120, label='test')

plt.legend()

plt.show()
X = df_t.drop(['ID_code', 'target'], axis=1)

y = df_t['target']

X_test = df_test.drop(['ID_code'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()

logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(y_test, y_pred)

cm = confusion_matrix

print(confusion_matrix)
from sklearn.metrics import classification_report

pred = logreg.predict(X_test)

print(classification_report(y_test, pred))
cm_sum = np.sum(cm, axis=1, keepdims=True)

cm_perc = cm / cm_sum.astype(float) * 100

annot = np.empty_like(cm).astype(str)

nrows, ncols = cm.shape

for i in range(nrows):

    for j in range(ncols):

        c = cm[i, j]

        p = cm_perc[i, j]

        if i == j:

            s = cm_sum[i]

            annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)

        elif c == 0:

            annot[i, j] = ''

        else:

            annot[i, j] = '%.1f%%\n%d' % (p, c)



cm = pd.DataFrame(cm, index=np.unique(y_test), columns=np.unique(y_test))

cm.index.name = 'Actual'

cm.columns.name = 'Predicted'



fig, ax = plt.subplots(figsize=[5,2])



sns.heatmap(cm, cmap= "YlGnBu", annot= annot, fmt='', ax=ax)
logist_pred = logreg.predict_proba(X_test)[:,1]

logist_pred
from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))

fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])

plt.figure()

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
def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))



train_pred = logreg.predict(X_train)

print('RMSLE : {:.4f}'.format(rmsle(y_train, train_pred)))
test_t = df_test.drop(['ID_code'], axis = 1)

logreg_pred_test = logreg.predict_proba(test_t)[:,1]

result = df_test[['ID_code']]

result['target'] = logreg_pred_test

result.head()



result.to_csv('log_reg_baseline.csv', index = False)
nb = GaussianNB()

nb.fit(X_train, y_train)

acc = nb.score(X_test,y_test)*100

y_pred = nb.predict(X_test)

cm_nb = confusion_matrix

print("Accuracy of Naive Bayes: {:.2f}%".format(acc))
print(classification_report(y_test,y_pred))
cm=confusion_matrix

cm_sum = np.sum(cm, axis=1, keepdims=True)

cm_perc = cm / cm_sum.astype(float) * 100

annot = np.empty_like(cm).astype(str)

nrows, ncols = cm.shape

for i in range(nrows):

    for j in range(ncols):

        c = cm[i, j]

        p = cm_perc[i, j]

        if i == j:

            s = cm_sum[i]

            annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)

        elif c == 0:

            annot[i, j] = ''

        else:

            annot[i, j] = '%.1f%%\n%d' % (p, c)



cm = pd.DataFrame(cm, index=np.unique(y_test), columns=np.unique(y_test))

cm.index.name = 'Actual'

cm.columns.name = 'Predicted'



fig, ax = plt.subplots(figsize=[5,2])



sns.heatmap(cm, cmap= "YlGnBu", annot= annot, fmt='', ax=ax)



sns.heatmap(cm, cmap= "YlGnBu", annot= annot, fmt='', ax=ax)
logit_roc_auc = roc_auc_score(y_test, nb.predict(X_test))

fpr, tpr, thresholds = roc_curve(y_test, nb.predict_proba(X_test)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('')

plt.legend(loc="lower right")

plt.savefig('Log_ROC')

plt.show()
nb_pred_test = nb.predict_proba(test_t)[:,1]

result = df_test[['ID_code']]

result['target'] = nb_pred_test

result.head()
result.to_csv('NB_baseline.csv', index = False)
tree = DecisionTreeClassifier(class_weight='balanced',max_depth=4)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
accuracy_score(y_test,y_pred)
print(classification_report(y_test,y_pred))
cm=confusion_matrix

cm_sum = np.sum(cm, axis=1, keepdims=True)

cm_perc = cm / cm_sum.astype(float) * 100

annot = np.empty_like(cm).astype(str)

nrows, ncols = cm.shape

for i in range(nrows):

    for j in range(ncols):

        c = cm[i, j]

        p = cm_perc[i, j]

        if i == j:

            s = cm_sum[i]

            annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)

        elif c == 0:

            annot[i, j] = ''

        else:

            annot[i, j] = '%.1f%%\n%d' % (p, c)



cm = pd.DataFrame(cm, index=np.unique(y_test), columns=np.unique(y_test))

cm.index.name = 'Actual'

cm.columns.name = 'Predicted'



fig, ax = plt.subplots(figsize=[5,2])



sns.heatmap(cm, cmap= "YlGnBu", annot= annot, fmt='', ax=ax)



sns.heatmap(cm, cmap= "YlGnBu", annot= annot, fmt='', ax=ax)
logit_roc_auc = roc_auc_score(y_test, tree.predict(X_test))

fpr, tpr, thresholds = roc_curve(y_test, tree.predict_proba(X_test)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('')

plt.legend(loc="lower right")

plt.savefig('Log_ROC')

plt.show()
tr_pred_test = tree.predict_proba(test_t)[:,1]

result = df_test[['ID_code']]

result['target'] =tr_pred_test

result.head()
result.to_csv('tree_baseline.csv', index = False)
from sklearn.ensemble import RandomForestClassifier



# Create the model with 100 trees

model = RandomForestClassifier(n_estimators=100, 

                               bootstrap = True,

                               max_features = 'sqrt')

# Fit on training data

model.fit(X_train, y_train)
# Use the forest's predict method on the test data

predictions = model.predict(X_test)

# Calculate the absolute errors

errors = abs(predictions - y_test)

# Print out the mean absolute error (mae)

print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
y_pred = model.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
cm
from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

logit_roc_auc = roc_auc_score(y_test_d, model.predict(X_test_d))

fpr, tpr, thresholds = roc_curve(y_test_d, model.predict_proba(X_test_d)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='Random Forest (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.savefig('Log_ROC')

plt.show()
rd_pred_test = model.predict_proba(test_t)[:,1]

result = df_test[['ID_code']]

result['target'] =rd_pred_test

result.head()
result.to_csv('rd_baseline.csv', index = False)
from xgboost import XGBClassifier

model = XGBClassifier()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy_score(y_test,y_pred)
print(classification_report(y_test,y_pred))
cm=confusion_matrix

cm_sum = np.sum(cm, axis=1, keepdims=True)

cm_perc = cm / cm_sum.astype(float) * 100

annot = np.empty_like(cm).astype(str)

nrows, ncols = cm.shape

for i in range(nrows):

    for j in range(ncols):

        c = cm[i, j]

        p = cm_perc[i, j]

        if i == j:

            s = cm_sum[i]

            annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)

        elif c == 0:

            annot[i, j] = ''

        else:

            annot[i, j] = '%.1f%%\n%d' % (p, c)



cm = pd.DataFrame(cm, index=np.unique(y_test), columns=np.unique(y_test))

cm.index.name = 'Actual'

cm.columns.name = 'Predicted'



fig, ax = plt.subplots(figsize=[5,2])

sns.heatmap(cm, cmap= "YlGnBu", annot= annot, fmt='', ax=ax)



sns.heatmap(cm, cmap= "YlGnBu", annot= annot, fmt='', ax=ax)
logit_roc_auc = roc_auc_score(y_test, model.predict(X_test))

fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('')

plt.legend(loc="lower right")

plt.savefig('Log_ROC')

plt.show()
xg_pred_test = model.predict_proba(test_t)[:,1]

result = df_test[['ID_code']]

result['target'] =xg_pred_test

result.head()
result.to_csv('xg_baseline.csv', index = False)