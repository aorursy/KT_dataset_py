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
import numpy as np
import random
import catboost
import time
import xgboost
TRAIN_FILE = "../input/train_lyc.csv"
TEST_FILE = "../input/test_lyc.csv"
SUBM_FILE = "res"
ENSEMBLE_COUNT = 1      # doesn't matter in case of using boosting
N_EPOCHS = 3            # doesn't matter in the same case
CLASS_BALANCING = False # amount of survived is less
def onehotenc(num, count):
    r = [0 for i in range(count)]
    r[num] = 1
    return r

def get_X(path, delt):
    f = open(path, "rt")
    t = f.read().split("\n")
    f.close()
    embarked = {"Q": 0, "S": 1, "C": 2, "": 3}
    cabin = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "T": 8}
    ticket = {"1": 1, "C": 2, "P": 3, "S": 4, "2": 5, "3": 6, "4": 7, "8": 8, "W": 9, "F": 10, "6": 11, "A": 12, "B": 13, "7": 14, "L": 15, "5": 16, "9": 17}
    X = []
    for line in t:
        if line == "":
            continue
        if line[0] == "P":
            continue
        cols = line.split(",")
        #print(cols)
        if len(cols) > 4:
            xcols = onehotenc(int(cols[2 - delt]), 4)              #Pclass
            
            xcols.append(1 if cols[5 - delt] == "male" else 0)     #Sex
            if cols[6 - delt] != "":
                xcols.append(float(cols[6 - delt]))                    #Age
            else:
                xcols.append(30.0)
            
            xcols.append(float(cols[7 - delt]))                    #Sibsp
            c = 0
            if cols[9 - delt] != "":
                c = ticket[cols[9 - delt][0]]
            xcols.extend(onehotenc(c, len(ticket) + 1))
            
            if cols[10 - delt] != "":
                xcols.append(float(cols[10 - delt]))                   #Fare
            else:
                xcols.append(0)
                

            c = 0
            if cols[11 - delt] != "":
                c = cabin[cols[11 - delt][0]]                      #Cabin
                
            xcols.extend(onehotenc(c, len(cabin)+1))
            xcols.extend(onehotenc(embarked[cols[12 - delt]], len(embarked)))
            c = 0
            if cols[11 - delt] != "":
                c = 1 
                if cols[11 - delt][0] == "S":
                    c = 2
            xcols.extend(onehotenc(c, 3))
            if CLASS_BALANCING:
                if delt == 0:
                    if cols[1] == "1":
                        X.append(xcols)
                        X.append(xcols)
            X.append(xcols)
    return np.array(X)

def get_Y(path):
    f = open(path, "rt")
    t = f.read().split("\n")
    f.close()
    dic = {"Q": 0, "S": 1, "C": 2, "": 3}
    y = []
    for line in t:
        if line == "":
            continue
        if line[0] == "P":
            continue
        cols = line.split(",")
        
        if len(cols) > 4:
            ycols = int(cols[1])
            if CLASS_BALANCING:
                if ycols == 1:
                    y.append(ycols)
                    y.append(ycols)
            y.append(ycols)
    return np.array(y)
def get_rpart(X, y, part=0.5):
    resX = []
    resY = []
    X_ = list(X)
    y_ = list(y)
    for i in range(round(len(X_) * part)):
        id = random.randint(0, len(X_) - 1)
        resX.append(X_[i])
        resY.append(y_[i])
        
    return np.array(resX), np.array(resY)

def get_part(X, y, test_size=0.1):
    l = len(X)
    return X[:round(l * (1 - test_size))], y[:round(l * (1 - test_size))], X[round(l * (1 - test_size)):], y[round(l * (1 - test_size)):]
X, y, X_tr, y_tr = get_part(get_X(TRAIN_FILE, 0), get_Y(TRAIN_FILE), 0.08)
print(len(X), len(y))
ens = []
for i in range(ENSEMBLE_COUNT):
    a = catboost.CatBoostClassifier(1600, logging_level="Silent")
    X_, y_ = get_rpart(X, y, 1.0)
    a.fit(X_, y_)
    ens.append(a)
    print("#", i, "done.")
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