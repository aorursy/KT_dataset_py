# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import confusion_matrix,auc,roc_auc_score

from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import matplotlib.pyplot as plt

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# https://www.kaggle.com/lovedeepsaini/fraud-detection-with-naive-bayes-classifier/notebook
df = pd.read_csv("../input/creditcard.csv")

df.describe()
print("Class as pie chart:")

fig, ax = plt.subplots(1, 1)

print(df.Class.value_counts())

ax.pie(df.Class.value_counts(),autopct='%1.1f%%', labels=['Genuine','Fraud'], colors=['yellowgreen','r'])
print('Time variable')

df['Time_Hr'] = df["Time"]/3600

print(df["Time_Hr"].tail(5))

fig, (ax1, ax2) = plt.subplots(2, 1, sharex = True, figsize=(6,3))

ax1.hist(df.Time_Hr[df.Class==0],bins=48,color='g',alpha=0.5)

ax1.set_title('Genuine')

ax2.hist(df.Time_Hr[df.Class==1],bins=48,color='r',alpha=0.5)

ax2.set_title('Fraud')

plt.xlabel('Time (hrs)')

plt.ylabel('# transactions')
fig, (ax3,ax4) = plt.subplots(2,1, figsize = (6,3), sharex = True)

ax3.hist(df.Amount[df.Class==0],bins=50,color='g',alpha=0.5)

ax4.hist(df.Amount[df.Class==1],bins=50,color='r',alpha=0.5)

ax3.set_yscale('log') # to see the tails

ax3.set_title('Genuine') # to see the tails

ax3.set_ylabel('# transactions')

ax4.set_yscale('log') # to see the tails

ax4.set_title('Fraud') # to see the tails

ax4.set_xlabel('Amount ($)')

ax4.set_ylabel('# transactions') 
from sklearn.preprocessing import StandardScaler

df['scaled_Amount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1,1))

df = df.drop(['Amount'],axis=1)

df.head(10)
#let us check correlations and shapes of those 25 principal components.

# Features V1, V2, ... V28 are the principal components obtained with PCA.

import seaborn as sns

import matplotlib.gridspec as gridspec

gs = gridspec.GridSpec(28, 1)

plt.figure(figsize=(6,28*4))

for i, col in enumerate(df[df.iloc[:,0:28].columns]):

    ax5 = plt.subplot(gs[i])

    sns.distplot(df[col][df.Class == 1], bins=50, color='r')

    sns.distplot(df[col][df.Class == 0], bins=50, color='g')

    ax5.set_xlabel('')

    ax5.set_title('feature: ' + str(col))

plt.show()
def split_data(df, drop_list):

    df = df.drop(drop_list,axis=1)

    print(df.columns)

    #test train split time

    from sklearn.model_selection import train_test_split

    y = df['Class'].values #target

    X = df.drop(['Class'],axis=1).values #features

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,

                                                    random_state=42, stratify=y)



    print("train-set size: ", len(y_train),

      "\ntest-set size: ", len(y_test))

    print("fraud cases in test-set: ", sum(y_test))

    return X_train, X_test, y_train, y_test
def get_predictions(clf, X_train, y_train, X_test):

    # create classifier

    clf = clf

    # fit it to training data

    clf.fit(X_train,y_train)

    # predict using test data

    y_pred = clf.predict(X_test)

    # Compute predicted probabilities: y_pred_prob

    y_pred_prob = clf.predict_proba(X_test)

    #for fun: train-set predictions

    train_pred = clf.predict(X_train)

    print('train-set confusion matrix:\n', confusion_matrix(y_train,train_pred)) 

    return y_pred, y_pred_prob
from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression
def print_scores(y_test,y_pred,y_pred_prob):

    print('test-set confusion matrix:\n', confusion_matrix(y_test,y_pred)) 

    print("recall score: ", recall_score(y_test,y_pred))

    print("precision score: ", precision_score(y_test,y_pred))

    print("f1 score: ", f1_score(y_test,y_pred))

    print("accuracy score: ", accuracy_score(y_test,y_pred))

    print("ROC AUC: {}".format(roc_auc_score(y_test, y_pred_prob[:,1])))
drop_list = []

X_train, X_test, y_train, y_test = split_data(df, drop_list)

y_pred, y_pred_prob = get_predictions(GaussianNB(), X_train, y_train, X_test)

print_scores(y_test,y_pred,y_pred_prob)
drop_list = ['V28','V27','V26','V25','V24','V23','V22','V20','V15','V13','V8']

X_train, X_test, y_train, y_test = split_data(df, drop_list)

y_pred, y_pred_prob = get_predictions(GaussianNB(), X_train, y_train, X_test)

print_scores(y_test,y_pred,y_pred_prob)
drop_list = ['Time_Hr','V28','V27','V26','V25','V24','V23','V22','V20','V15','V13','V8']

X_train, X_test, y_train, y_test = split_data(df, drop_list)

y_pred, y_pred_prob = get_predictions(GaussianNB(), X_train, y_train, X_test)

print_scores(y_test,y_pred,y_pred_prob)
df = df.drop(drop_list,axis=1)

print(df.columns)