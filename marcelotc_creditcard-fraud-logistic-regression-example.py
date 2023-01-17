import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import math

import sklearn

import numpy as np 



import warnings

warnings.filterwarnings('ignore')



%matplotlib inline
df = pd.read_csv("../input/creditcardfraud/creditcard.csv")
print('Total de linhas e colunas\n\n',df.shape,'\n')
df.isnull().sum()
df.info()
df.describe().round()
print ('Not Fraud % ',round(df['Class'].value_counts()[0]/len(df)*100,2))

print ()

print (round(df.Amount[df.Class == 0].describe(),2))

print ()

print ()

print ('Fraud %    ',round(df['Class'].value_counts()[1]/len(df)*100,2))

print ()

print (round(df.Amount[df.Class == 1].describe(),2))
plt.figure(figsize=(10,8))

sns.set_style('darkgrid')

sns.barplot(x=df['Class'].value_counts().index,y=df['Class'].value_counts(), palette=["C1", "C8"])

plt.title('Non Fraud X Fraud')

plt.ylabel('Count')

plt.xlabel('0: Non Fraud,  1: Fraud')

print ('Non Fraud % ',round(df['Class'].value_counts()[0]/len(df)*100,2))

print ('Fraud %    ',round(df['Class'].value_counts()[1]/len(df)*100,2));
feature_names = df.iloc[:, 1:30].columns

target = df.iloc[:1, 30:].columns



data_features = df[feature_names]

data_target = df[target]
feature_names
target
from sklearn.model_selection import train_test_split

np.random.seed(123)

X_train, X_test, y_train, y_test = train_test_split(data_features, data_target, 

                                                    train_size = 0.70, test_size = 0.30, random_state = 1)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train, y_train)
def PrintStats(cmat, y_test, pred):

    tpos = cmat[0][0]

    fneg = cmat[1][1]

    fpos = cmat[0][1]

    tneg = cmat[1][0]
def RunModel(model, X_train, y_train, X_test, y_test):

    model.fit(X_train, y_train.values.ravel())

    pred = model.predict(X_test)

    matrix = confusion_matrix(y_test, pred)

    return matrix, pred
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve

import scikitplot as skplt
cmat, pred = RunModel(lr, X_train, y_train, X_test, y_test)
import scikitplot as skplt

skplt.metrics.plot_confusion_matrix(y_test, pred)
accuracy_score(y_test, pred)
print (classification_report(y_test, pred))
# The function "len" counts the number of classes = 1 and saves it as an object "fraud_records"

fraud_records = len(df[df.Class == 1])



# Defines the index for fraud and non-fraud in the lines:

fraud_indices = df[df.Class == 1].index

not_fraud_indices = df[df.Class == 0].index



# Randomly collect equal samples of each type:

under_sample_indices = np.random.choice(not_fraud_indices, fraud_records, False)

df_undersampled = df.iloc[np.concatenate([fraud_indices, under_sample_indices]),:]

X_undersampled = df_undersampled.iloc[:,1:30]

Y_undersampled = df_undersampled.Class

X_undersampled_train, X_undersampled_test, Y_undersampled_train, Y_undersampled_test = train_test_split(X_undersampled, Y_undersampled, test_size = 0.30)
lr_undersampled = LogisticRegression()

cmat, pred = RunModel(lr_undersampled, X_undersampled_train, Y_undersampled_train, X_undersampled_test, Y_undersampled_test)

PrintStats(cmat, Y_undersampled_test, pred)
skplt.metrics.plot_confusion_matrix(Y_undersampled_test, pred)
accuracy_score(Y_undersampled_test, pred)
print (classification_report(Y_undersampled_test, pred))
lr_undersampled = LogisticRegression()

cmat, pred = RunModel(lr_undersampled, X_undersampled_train, Y_undersampled_train, X_test, y_test)

PrintStats(cmat, y_test, pred)
skplt.metrics.plot_confusion_matrix(y_test, pred)
accuracy_score(y_test, pred)
print (classification_report(y_test, pred))
from sklearn.model_selection import GridSearchCV



param_grid = {"C": [1,2,3,4,5,6,7,8,9,10], 

              "penalty": ['l1','l2']} #Parameters

            

grid_search = GridSearchCV(lr, param_grid, scoring="precision") #score

grid_search.fit(y_test, pred)



lr = grid_search.best_estimator_ 

grid_search.best_params_, grid_search.best_score_
lr_undersampled = LogisticRegression(C=1, penalty='l2')

cmat, pred = RunModel(lr_undersampled, X_undersampled_train, Y_undersampled_train, X_undersampled_test, Y_undersampled_test)

PrintStats(cmat, Y_undersampled_test, pred)
skplt.metrics.plot_confusion_matrix(Y_undersampled_test, pred)
accuracy_score(Y_undersampled_test, pred)
print (classification_report(Y_undersampled_test, pred))
lr = LogisticRegression(C=1, penalty='l2')

cmat, pred = RunModel(lr, X_undersampled_train, Y_undersampled_train, X_test, y_test)

PrintStats(cmat, y_test, pred)
skplt.metrics.plot_confusion_matrix(y_test, pred)
accuracy_score(y_test, pred)
print (classification_report(y_test, pred))
from sklearn import metrics                           
clf = LogisticRegression(C=1, penalty='l2')

clf.fit(X_undersampled_train, Y_undersampled_train)

y_pred = clf.predict(X_test)



y_pred_probability = clf.predict_proba(X_test)[::,1]

fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_probability)

auc = metrics.roc_auc_score(y_test, pred)

plt.plot(fpr,tpr,label="LogisticRegression, auc="+str(auc))

plt.legend(loc=4)

plt.show()