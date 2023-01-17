import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier, MLPRegressor

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

from sklearn.metrics import mean_squared_error,confusion_matrix, precision_score, recall_score, auc,roc_curve
def f(s):

    if s=="negative":

        return 0

    else:

        return 1
def f1(s):

    if s=="M":

        return 2

    else:

        return 1
df=pd.read_csv("../input/project/pima.csv")

df["Class"]=df.Class.apply(f)

#df["Sex"]=df.Sex.apply(f)

N=df.shape[0]

M=df.shape[1]

x=df.values[:, :M-1]

y=df.values[:, M-1]

scaler=preprocessing.StandardScaler()

x[0], y[0]
#N = 9

#M = 1484

#b=np.array([[x[i][j]**2 for j in range(N-1)] for i in range(M)])

#c=np.array([[x[i][j]*x[i][(j+1)%(N-1)] for j in range(N-1)] for i in range(M)])

#d=np.array([[x[i][j]*x[i][(j+2)%(N-1)] for j in range(N-1)] for i in range(M)])

#e=np.array([[x[i][j]*x[#i][(j+3)%(N-1)] for j in range(N-1)] for i in range(M)])



#x=np.concatenate((x, b),axis=1)

#x=np.concatenate((x, c),axis=1)

#x=np.concatenate((x, d),axis=1)

#x=np.concatenate((x, e),axis=1)

x_train_whole, x_test_whole, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)

#scaler.fit(x_train_whole)

#scaler.transform(x_train_whole)

#scaler.transform(x_test_whole)

x.shape
clf = MLPRegressor(activation='logistic', solver='lbfgs', random_state=1, alpha=0.05, learning_rate_init=0.5, hidden_layer_sizes=(5, 2,), max_iter=3000)
x_train = x_train_whole[:, [0, 2, 4, 6]]

x_test = x_test_whole[:, [0, 2, 4, 6]]

clf.fit(x_train, y_train)

y_train_pred=clf.predict(x_train)

y_test_pred=clf.predict(x_test)

x_new_train=y_train_pred

x_new_test=y_test_pred

fp, tp, th = roc_curve(y_test, y_test_pred)

#precision_score(y_test, y_test_pred), recall_score(y_test, y_test_pred), auc(fp, tp)
x_train = x_train_whole[:, [1, 3, 5, 7]]

x_test = x_test_whole[:, [1, 3, 5, 7]]

clf.fit(x_train, y_train)

y_train_pred=clf.predict(x_train)

y_test_pred=clf.predict(x_test)

x_new_train=np.column_stack((x_new_train, y_train_pred))

x_new_test=np.column_stack((x_new_test, y_test_pred))

fp, tp, th = roc_curve(y_test, y_test_pred)

#precision_score(y_test, y_test_pred), recall_score(y_test, y_test_pred), auc(fp, tp)
x_train = x_train_whole[:, [0, 1, 2, 3]]

x_test = x_test_whole[:, [0, 1, 2, 3]]

clf.fit(x_train, y_train)

y_train_pred=clf.predict(x_train)

y_test_pred=clf.predict(x_test)

x_new_train=np.concatenate((x_new_train, y_train_pred.reshape(-1,1)), axis=1)

x_new_test=np.concatenate((x_new_test, y_test_pred.reshape(-1,1)), axis=1)

fp, tp, th = roc_curve(y_test, y_test_pred)

#precision_score(y_test, y_test_pred), recall_score(y_test, y_test_pred), auc(fp, tp)
x_train = x_train_whole[:, [4, 5, 6, 7]]

x_test = x_test_whole[:, [4, 5, 6, 7]]

clf.fit(x_train, y_train)

y_train_pred=clf.predict(x_train)

y_test_pred=clf.predict(x_test)

x_new_train=np.concatenate((x_new_train, y_train_pred.reshape(-1,1)), axis=1)

x_new_test=np.concatenate((x_new_test, y_test_pred.reshape(-1,1)), axis=1)

fp, tp, th = roc_curve(y_test, y_test_pred)

#precision_score(y_test, y_test_pred), recall_score(y_test, y_test_pred), auc(fp, tp)
x_new_train
clf = MLPClassifier(activation='logistic', solver='lbfgs', random_state=1, alpha=0.05, learning_rate_init=0.5, hidden_layer_sizes=(5, 2,), max_iter=3000)
clf.fit(x_new_train, y_train)

y_final_train_pred=clf.predict(x_new_train)

y_final_test_pred=clf.predict(x_new_test)

fp, tp, th = roc_curve(y_test, y_final_test_pred)

precision_score(y_test, y_final_test_pred), recall_score(y_test, y_final_test_pred), auc(fp, tp)
x_train = x_train_whole

x_test = x_test_whole

clf.fit(x_train, y_train)

y_train_pred=clf.predict(x_train)

y_test_pred=clf.predict(x_test)

fp, tp, th = roc_curve(y_test, y_test_pred)

precision_score(y_test, y_test_pred), recall_score(y_test, y_test_pred), auc(fp, tp)