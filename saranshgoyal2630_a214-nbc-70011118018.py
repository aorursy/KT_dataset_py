# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix,auc,roc_auc_score

from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score
df = pd.read_csv( "../input/customer-behaviour/Customer_Behaviour.csv")
df.head(15)

print(df.describe())

print(df.info())
print("Class as pie chart:")

fig, ax = plt.subplots(1, 1)

ax.pie(df.Purchased.value_counts(),autopct='%1.1f%%', labels=['Not Purchased','Purchased'], colors=['r','yellowgreen',])

plt.axis('equal')

plt.ylabel('')
print("Age Variable")

fig, (ax1, ax2) = plt.subplots(2, 1, sharex = True, figsize=(6,3))

ax1.hist(df.Age[df.Purchased==0],bins=48,color='r',alpha=0.5)

ax1.set_title('Not Purchased')

ax2.hist(df.Age[df.Purchased==1],bins=48,color='g',alpha=0.5)

ax2.set_title('Purchased')

plt.xlabel('Age')

plt.ylabel('# transactions')
fig, (ax3,ax4) = plt.subplots(2,1, figsize = (6,3), sharex = True)

ax3.hist(df.EstimatedSalary[df.Purchased==0],bins=50,color='r',alpha=0.5)

ax3.set_yscale('log') # to see the tails

ax3.set_title('Not Purchased') # to see the tails

ax3.set_ylabel('# transactions')

ax4.hist(df.EstimatedSalary[df.Purchased==1],bins=50,color='g',alpha=0.5)

ax4.set_yscale('log') # to see the tails

ax4.set_title('Purchased') # to see the tails

ax4.set_xlabel('EstimatedSalary ($)')

ax4.set_ylabel('# transactions')
def split_data(df, drop_list):

    df = df.drop(drop_list,axis=1)

    print(df.columns)

    

    from sklearn.model_selection import train_test_split

    y = df['Purchased'].values #target

    X = df.drop(['Purchased'],axis=1).values #features

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,

                                                    random_state=42, stratify=y)



    print("train-set size: ", len(y_train),

      "\ntest-set size: ", len(y_test))

    print("Purchased: ", sum(y_test))

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
def print_scores(y_test,y_pred,y_pred_prob):

    print('test-set confusion matrix:\n', confusion_matrix(y_test,y_pred)) 

    print("recall score: ", recall_score(y_test,y_pred))

    print("precision score: ", precision_score(y_test,y_pred))

    print("f1 score: ", f1_score(y_test,y_pred))

    print("accuracy score: ", accuracy_score(y_test,y_pred))

    print("ROC AUC: {}".format(roc_auc_score(y_test, y_pred_prob[:,1])))
from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression
drop_list = ['Gender']

X_train, X_test, y_train, y_test = split_data(df,drop_list)

y_pred, y_pred_prob = get_predictions(GaussianNB(), X_train, y_train, X_test)

print_scores(y_test,y_pred,y_pred_prob)