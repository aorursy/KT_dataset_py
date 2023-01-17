import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

data = pd.read_csv('/kaggle/input/personal-loan/Bank_Personal_Loan_Modelling-1.xlsx')
data.info()
data.head(2)
X = data.drop(['Personal Loan'], axis = 1)

y = data['Personal Loan']
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42) 

#Random_state = 42 is to be tried later
from sklearn.metrics import roc_auc_score

from sklearn.metrics import confusion_matrix



def clf_scores(clf, y_predicted):

    # Accuracy

    acc_train = clf.score(X_train, y_train)*100

    acc_test = clf.score(X_test, y_test)*100

    

    roc = roc_auc_score(y_test, y_predicted)*100 

    tn, fp, fn, tp = confusion_matrix(y_test, y_predicted).ravel()

    cm = confusion_matrix(y_test, y_predicted)

    correct = tp + tn

    incorrect = fp + fn

    

    return acc_train, acc_test, roc, correct, incorrect, cm
#1. Logistic regression



from sklearn.linear_model import LogisticRegression

clf_lr = LogisticRegression()

clf_lr.fit(X_train, y_train)



Y_pred_lr = clf_lr.predict(X_test)

print(clf_scores(clf_lr, Y_pred_lr))



# without random state - (90.97142857142858, 90.4, 62.7838637454022), 1350
#2. KNN



from sklearn.neighbors import KNeighborsClassifier

clf_knn = KNeighborsClassifier(n_neighbors=3)

clf_knn.fit(X_train, y_train)



Y_pred_knn = clf_knn.predict(X_test)

print(clf_scores(clf_knn, Y_pred_knn))

#1348
#3. Naive Bayes



from sklearn.naive_bayes import GaussianNB

clf_gnb = GaussianNB()

clf_gnb.fit(X_train, y_train)



Y_pred_gnb = clf_gnb.predict(X_test)

print(clf_scores(clf_gnb, Y_pred_gnb))

#4. Decision tree

from sklearn.tree import DecisionTreeClassifier



clf_dt = DecisionTreeClassifier(random_state=0)

clf_dt.fit(X_train, y_train)

clf_dt.fit(X_train, y_train)



Y_pred_dt = clf_dt.predict(X_test)

print(clf_scores(clf_dt, Y_pred_dt))
#5. Radom forest classifier



from sklearn.ensemble import RandomForestClassifier



clf_rfc = RandomForestClassifier(max_depth=10, random_state=42)

clf_rfc.fit(X_train, y_train)



Y_pred_rfc = clf_rfc.predict(X_test)

print(clf_scores(clf_rfc, Y_pred_rfc))

#6. Gradient boosting classifier



from sklearn.ensemble import GradientBoostingClassifier



clf_gbc = GradientBoostingClassifier(random_state=42)

clf_gbc.fit(X_train, y_train)



Y_pred_gbc = clf_gbc.predict(X_test)

print(clf_scores(clf_gbc, Y_pred_gbc))