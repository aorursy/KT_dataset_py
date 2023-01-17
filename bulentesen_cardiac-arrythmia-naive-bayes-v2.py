from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
print(os.listdir('../input'))
veri = pd.read_csv('../input/data_arrhythmia.csv', delimiter=';')

veri.dataframeName = 'data_arrhythmia.csv'

nRow, nCol = veri.shape

print(f'Bu verisetinde {nRow} satır ve {nCol} Sutun vardır')
veri.head()
veri.drop(["J","R'_wave","S'_wave", "AB", "AC", "AD","AE", "AF", "AG", "AL", "AN", "AO", "AP", "AR", "AS", "AT", "AZ", "AB'", "BC", "BD", "BE", "BG", "BH", "BP", "BR", "BS", "BT", "BU", 

          "CA", "CD", "CE", "Cf", "CG", "CH", "CI", "CM","CN","CP","CR","CS","CT","CU","CV","DE","DF","DG","DH","DI","DJ","DR","DS","DT","DU","DV","DY","EG",

          "EH", "EL", "ER", "ET", "EU", "EV", "EY", "EZ", "FA", "FE", "FF", "FH", "FI", "FJ", "FK", "FL", "FM", "FR", "FS", "FU", "FV", "FY", "FZ", "GA",

          "GB", "GG", "GH", "HD", "HE", "HO", "IA", "IB", "IK", "IL", "IY", "JI", "JS", "JT", "KF", "KO", "KP", "LB", "LC", "T", "P", "QRST", "heart_rate"], axis=1, inplace=True)
veri.head()
y = veri.diagnosis.values

x = veri.drop(["diagnosis"],axis=1)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=42)

x_train, x_validate, y_train, y_validate = train_test_split(x_train,y_train,test_size=0.25, random_state=42)
x_train.head()
x_test.head()
print("x_train: ",x_train.shape)

print("y_train: ",y_train.shape)

print("x_test: ",x_test.shape)

print("y_test: ",y_test.shape)

print("x_validate: ",x_test.shape)

print("y_validate: ",y_test.shape)
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train,y_train)
y_pred = nb.predict(x_test)

print("Tahmin Edilen Deger: ",y_pred)
from sklearn import metrics

from sklearn.metrics import confusion_matrix

from sklearn.metrics import f1_score

from sklearn.metrics import precision_score

print("Confussion Matrisi: ",confusion_matrix(y_test, y_pred))

print("Dogruluk oranı: ",metrics.accuracy_score(y_test,y_pred))

print("f1 Score: ",f1_score(y_test, y_pred, average='macro'))

print("Kesinlik Değeri: ",precision_score(y_test, y_pred, average='macro'))
