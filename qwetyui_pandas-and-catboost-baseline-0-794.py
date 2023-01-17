''' # you might need sklearn to replace my model with sklearn one
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
'''
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
import numpy as np
import random
import catboost
import time
import xgboost
import pandas as pd
TRAIN_FILE = "../input/train_lyc.csv"
TEST_FILE = "../input/test_lyc.csv"
SUBM_FILE = "res"
ENSEMBLE_COUNT = 5      # doesn't matter in case of using boosting
N_EPOCHS = 3            # doesn't matter in the same case
def onehotenc(num, count):
    r = [0 for i in range(count)]
    r[num] = 1
    return r

def ohe_dict(kostya_lox, dic, label):
    for key in dic:
        kostya_lox[label + "_" + key] = kostya_lox[label].apply(lambda x: 1 if len(str(x)) > 0 and str(x)[0] == key else 0)

def get_X(path, delt):
    kostya_lox = pd.read_csv(path)
    kostya_lox = kostya_lox[["Sex", "Pclass", "Age", "Parch", "SibSp", "Cabin", "Ticket", "Fare", 'Embarked']]
    embarked = {"Q": 0, "S": 1, "C": 2, "": 3}
    cabin = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "T": 8}
    ticket = {"1": 1, "C": 2, "P": 3, "S": 4, "2": 5, "3": 6, "4": 7, "8": 8, "W": 9, "F": 10, "6": 11, "A": 12, "B": 13, "7": 14, "L": 15, "5": 16, "9": 17}
    sex = {"m": 0, "f": 1}
    pclass = {"1": 0, "2": 1, "3": 2}
    ohe_dict(kostya_lox, embarked, "Embarked")
    ohe_dict(kostya_lox, sex, "Sex")
    ohe_dict(kostya_lox, cabin, "Cabin")
    ohe_dict(kostya_lox, ticket, "Ticket")
    ohe_dict(kostya_lox, pclass, "Pclass")
    kostya_lox = kostya_lox.drop(["Embarked", "Sex", "Cabin", "Ticket", "Pclass"], axis=1)
    return kostya_lox.astype(np.float64)

def get_Y(path):
    kostya_lox = pd.read_csv(path)
    kostya_lox = kostya_lox["Survived"]
    return kostya_lox.astype(np.float64)
Xf, yf = get_X(TRAIN_FILE, 0), get_Y(TRAIN_FILE)
Xf.head()
yf.head()
Xf, yf = np.array(Xf), np.array(yf)
ens = []
print("Fitting...")
for i in range(ENSEMBLE_COUNT):
    a = catboost.CatBoostClassifier(1000, logging_level="Silent")
    X, X_tr, y, y_tr = train_test_split(Xf, yf, test_size=0.2)
    a.fit(X, y)
    ens.append(a)
    print(round(100 if ENSEMBLE_COUNT == 1 else i / (ENSEMBLE_COUNT - 1) * 100), "% done.")

print("Ready.")
def predict(X):
    y = 0.0
    for model in ens:
        y += model.predict(X)
    return y / len(ens)
y_pred = [int(i) for i in list(predict(X))]
y_tr_pred = [int(i) for i in list(predict(X_tr))]
def count_tftnrel(y_true, y_pred):
    TP = 0
    TN = 0
    P = 0
    N = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            if y_true[i] == 1:
                TP += 1
            else:
                TN += 1
        if y_true[i] == 1:
            P += 1
        else:
            N += 1
    if P * N == 0:
        return 2, 2
    else:
        return (TP + TN) / (P + N), TP * TN / P / N, TP / 2 / P + TN / 2 / N
tftn = count_tftnrel(list(y), y_pred)
tftn_tr = count_tftnrel(list(y_tr), y_tr_pred)
print("Train acc      :", round((np.array(y_pred) == y).mean()*100, 1), "%")
print("Train TF*TN acc:", round(tftn[0], 3))
print("Train TF+TN acc:", round(tftn[1], 3))
print("Train avg acc  :", round(tftn[2], 3))
print("")
tacc = round((np.array(y_tr_pred) == y_tr).mean()*100, 1)
print("Test acc       :", tacc, "%")
print("Test TF*TN acc :", round(tftn_tr[0], 3))
print("Test TF+TN acc :", round(tftn_tr[1], 3))
print("Test avg acc   :", round(tftn_tr[2], 3))
X_test = get_X(TEST_FILE, 1)
answs = list(predict(X_test))
text = "PassengerId,Survived\n"
for i in range(len(answs)):
    text += str(i + 500) + "," + str(int(answs[i])) + "\n"
f = open(SUBM_FILE + str(round(time.time())) + ".csv", "wt")
f.write(text)
f.close()