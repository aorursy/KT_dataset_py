import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

#from sklearn.naive_bayes import GaussianNB

#from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import IsolationForest

from sklearn.neighbors import LocalOutlierFactor
df = pd.read_csv(r'../input/creditcardfraud/creditcard.csv')

df['Amount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))

#df

df.describe().transpose()
df.isnull().sum(axis = 0) #checking null values in the dataset
colors = ["b", "r"]

sns.countplot('Class', data=df, palette=colors)
#Creating a sample dataset

df2= df.sample(frac = 0.2,random_state=1)



df2.shape
#Correlation

corr = df2.corr()

sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 2.5})

plt.figure(figsize=(20,20))

# create a mask so we only see the correlation values once

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask, 1)] = True

a = sns.heatmap(corr,mask=mask, annot=True, fmt='.2f')

rotx = a.set_xticklabels(a.get_xticklabels(), rotation=90)

roty = a.set_yticklabels(a.get_yticklabels(), rotation=30)
# We Will check Do fraudulent transactions occur more often during certain time frame and visualize the data.

fraud_transactions2 = df2[df2.Class == 1]

Norm_transactions2 = df2[df2.Class == 0]

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True,figsize=(10,10))

f.suptitle('Time of transaction vs Amount by class')

ax1.scatter(fraud_transactions2.Time, fraud_transactions2.Amount)

ax1.set_title('Fraud')

ax2.scatter(Norm_transactions2.Time, Norm_transactions2.Amount)

ax2.set_title('Normal')

plt.xlabel('Time (in Seconds)')

plt.ylabel('Amount')

plt.show()

#fraud_transactions
X = df.drop(['Class'],axis=1).values

y = df[ 'Class'].values

state = np.random.RandomState(42)

# X_outliers = state.uniform(low=0, high=1, size=(X.shape[0], X.shape[1]))
X2 = df2.drop(['Class'],axis=1).values

y2 = df2[ 'Class'].values
from sklearn.linear_model import LogisticRegression

X_train2, X_test2, y_train2, y_test2 = train_test_split(X2,y2, test_size=.35,random_state=123)

y_train2 = np.ravel(y_train2)

lg=LogisticRegression()

lg.fit(X_train2,y_train2)

y_pred2=lg.predict(X_test2)

print(classification_report(y_test2,y_pred2))

cnf_matrix=(confusion_matrix(y_test2,y_pred2))

print(cnf_matrix)

print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))

print(accuracy_score(y_test2,y_pred2))

X = df.drop(['Class'],axis=1).values

y = df[ 'Class'].values

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.35,random_state=123)

y_train = np.ravel(y_train)

lg=LogisticRegression()

lg.fit(X_train,y_train)

y_pred=lg.predict(X_test)

print(classification_report(y_test,y_pred))

cnf_matrix=(confusion_matrix(y_test,y_pred))

print(cnf_matrix)

#print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))

print(accuracy_score(y_test,y_pred))
outlier_fraction2 = len(fraud_transactions2)/float(len(Norm_transactions2))

IF2= IsolationForest(n_estimators=100, max_samples=len(X2),\

                                       contamination=outlier_fraction2,random_state=state, verbose=0)

IF2.fit(X2)

y_pred2=IF2.predict(X2)

scores_prediction = IF2.decision_function(X2)

y_pred2[y_pred2 == 1] = 0

y_pred2[y_pred2 == -1] = 1

n_errors = (y_pred2 != y2).sum()

print("Isolation Factor is: {}".format(n_errors))

print("Accuracy score is: {}".format(accuracy_score(y2,y_pred2)))

#print(accuracy_score(y2,y_pred2))

print(classification_report(y2,y_pred2))

cnf_matrix=(confusion_matrix(y2,y_pred2))

print(cnf_matrix)

#print(n_errors) 

#y_pred2

fraud_transactions = df[df.Class == 1]

Norm_transactions = df[df.Class == 0]

outlier_fraction = len(fraud_transactions)/float(len(Norm_transactions))

IF= IsolationForest(n_estimators=100, max_samples=len(X),\

                                       contamination=outlier_fraction,random_state=state, verbose=0)

IF.fit(X)

y_pred=IF.predict(X)

scores_prediction = IF.decision_function(X)

y_pred[y_pred == 1] = 0

y_pred[y_pred == -1] = 1

n_errors = (y_pred != y).sum()

print("Isolation Factor is: {}".format(n_errors))

print("Accuracy score is: {}".format(accuracy_score(y,y_pred)))

#print(accuracy_score(y2,y_pred2))

print(classification_report(y,y_pred))

cnf_matrix=(confusion_matrix(y,y_pred))

print(cnf_matrix)

#print(n_errors) 

#y_pred2

LOF =LocalOutlierFactor(n_neighbors=20, algorithm='auto', 

                                              leaf_size=30, metric='minkowski',

                                              p=2, metric_params=None, contamination=outlier_fraction)

y_pred3 = LOF.fit_predict(X2)

scores_prediction = LOF.negative_outlier_factor_

y_pred3[y_pred3 == 1] = 0

y_pred3[y_pred3 == -1] = 1

print("Local outlier Factor is: {}".format(n_errors))

print("Accuracy score is: {}".format(accuracy_score(y2,y_pred3)))

print(classification_report(y2,y_pred3))

cnf_matrix=(confusion_matrix(y2,y_pred3))

print(cnf_matrix)

n_errors = (y_pred3 != y2).sum()

 
LOF2 =LocalOutlierFactor(n_neighbors=20, algorithm='auto', 

                                              leaf_size=30, metric='minkowski',

                                              p=2, metric_params=None, contamination=outlier_fraction)

y_pred3 = LOF2.fit_predict(X)

scores_prediction = LOF2.negative_outlier_factor_

y_pred3[y_pred3 == 1] = 0

y_pred3[y_pred3 == -1] = 1

print("Local outlier Factor is: {}".format(n_errors))

print("Accuracy score is: {}".format(accuracy_score(y,y_pred3)))

print(classification_report(y,y_pred3))

cnf_matrix=(confusion_matrix(y,y_pred3))

print(cnf_matrix)

n_errors = (y_pred3 != y).sum()