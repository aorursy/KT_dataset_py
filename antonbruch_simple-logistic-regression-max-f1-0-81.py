import pandas as pd

import numpy as np



from sklearn import linear_model

from sklearn.model_selection import train_test_split



import seaborn as sns
data = pd.read_csv("../input/creditcard.csv")

data.head()
data.shape
X = data[data.columns[0:30]]

y = data["Class"]
X.shape
lr = linear_model.LogisticRegression()

lr.fit(X, y)

print(lr.score(X, y))
from sklearn.metrics import f1_score, average_precision_score, precision_score, recall_score
y_pred=lr.predict(X)

print("Accuracy:", lr.score(X, y))

print("Precision:", precision_score(y, y_pred))

print("Recall:", recall_score(y, y_pred))

print("F1:", f1_score(y, y_pred))

print("Area under precision Recall:", average_precision_score(y, y_pred))

from sklearn.model_selection import cross_val_score, train_test_split
X_Legit=data.query("Class==0").drop(["Amount","Class"],1)

y_Legit=data.query("Class==0")["Class"]

X_Fraud=data.query("Class==1").drop(["Amount","Class"],1)

y_Fraud=data.query("Class==1")["Class"]

#split data into training and cv set

X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X_Legit, y_Legit, test_size=0.3)

X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_Fraud, y_Fraud, test_size=0.3)

X_test = X_test_l.append(X_test_f)

y_test = y_test_l.append(y_test_f)

X_train = X_train_l.append(X_train_f)

y_train = y_train_l.append(y_train_f)
def cv_run(X_train, X_test, y_train, y_test):

    bestC = 1.

    bestScore = 0.

    for C in [0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]:

        lr = linear_model.LogisticRegression(C=C)

        lr.fit(X_train, y_train)

        y_pred_train=lr.predict(X_train)

        score = f1_score(y_train, y_pred_train)

        if score > bestScore:

            bestC = C

            bestScore = score

    print( "Best C:", bestC)



    lr = linear_model.LogisticRegression(C=bestC)

    lr.fit(X_train, y_train)

    y_pred_test=lr.predict(X_test)

    y_pred_train=lr.predict(X_train)

    print("Train score:", lr.score(X_train, y_train))

    print("Test score:", lr.score(X_test, y_test))

    print("Train F1:", f1_score(y_train, y_pred_train))

    print("Test F1:", f1_score(y_test, y_pred_test))
cv_run(X_train, X_test, y_train, y_test)
from sklearn import preprocessing
X_test_scaled = preprocessing.scale(X_test)

X_train_scaled = preprocessing.scale(X_train)
cv_run(X_train_scaled, X_test_scaled, y_train, y_test)
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(2)

X_train_poly = poly.fit_transform(X_train_scaled)

X_test_poly = poly.fit_transform(X_test_scaled) 
def lr_run(X_train, X_test, y_train, y_test):

    lr = linear_model.LogisticRegression()

    lr.fit(X_train, y_train)

    y_test_pred=lr.predict(X_test)

    print("Accuracy on training set:", lr.score(X_train, y_train))

    print("Accuracy on test set:", lr.score(X_test, y_test))

    print("Precision on test set:", precision_score(y_test, y_test_pred))

    print("Recall on test set:", recall_score(y_test, y_test_pred))

    print("F1 on test set:", f1_score(y_test, y_test_pred))

    print("Area under precision Recall on test set:", average_precision_score(y_test, y_test_pred))

lr_run(X_train_poly, X_test_poly, y_train, y_test)
#poly = PolynomialFeatures(2)

#X_poly = poly.fit_transform(X_scaled) 

#cv_run(X_poly, y_scaled)
X_train_wo_t = np.delete(X_train_scaled, 0, 1)

X_test_wo_t = np.delete(X_test_scaled, 0, 1)
poly = PolynomialFeatures(2)

X_train_wo_t_poly = poly.fit_transform(X_train_wo_t)

X_test_wo_t_poly = poly.fit_transform(X_test_wo_t) 
lr_run(X_train_wo_t_poly, X_test_wo_t_poly, y_train, y_test)