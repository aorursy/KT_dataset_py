# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import collections
#from datetime import datetime
from datetime import date
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
import sklearn.metrics
dirnames=[]
for dirname, _, filenames in os.walk('/kaggle/input'):
    if "our_world_in_data" in dirname:
        for filename in filenames:
            #if ("statistics-and-research (1)" in dirname)or("testing-latest-data" in dirname):
            dirnames.append(os.path.join(dirname, filename))
            #print(os.path.join(dirname, filename))
print(dirnames)
print(dirnames[2],"\t",dirnames[-1])
data1=pd.read_csv(dirnames[1])
data2=pd.read_csv(dirnames[2])
data3=pd.read_csv(dirnames[3])#si
print(np.shape(data1))
print(np.shape(data2))
print(np.shape(data3))
counter=collections.Counter(data3["location"])       
print(np.shape(list(counter.keys())))
countries=list(counter.keys())
f=list(counter.values())
#it's for checking the minimum amount of data needed
print(np.sort(f))
for i in range(len(f)):
    if(f[i]==21):
        print(countries[i])
n_data=21
data3
# print(data3["date"][:5])
# a=date.fromisoformat(data3["date"][0])
# b=date.fromisoformat(data3["date"][1])
# print(date.toordinal(a))
# print(date.toordinal(b))
temp=[]
t_min=date.toordinal(date.fromisoformat(data3["date"][0]))
for i in range(1,len(data3["date"])):
        t_min=min([date.toordinal(date.fromisoformat(data3["date"][i])),t_min])
        #dates.append(date.toordinal(date.fromisoformat(data3["date"][j])))
print(t_min)
print(date.fromordinal(t_min))
#las categorías son los meses del año
import datetime
Y2=[]
for i in range(len(data3["date"])):
    datee = datetime.datetime.strptime(data3["date"][i], "%Y-%m-%d")
    Y2=np.append(Y2,datee.month+datee.year-2019)
#another y that will be tested
Y2=[0]
F=0
cnt=0
for i in range(len(f)):
    F+=f[i]
    if(i>0):
        cnt=0
        for j in range(F_last,F):
            if(np.abs(data3["total_cases"][i-1]-data3["total_cases"][i])<1e-4):
                cnt+=1
            if(cnt>3):
                Y2.append(1)
            else:
                Y2.append(0)
        F_last=F
print(Y2)
F=0
F_last=0
plt.figure()
for i in range(len(f)-2):
    F+=f[i]
    dates=[]
    for j in range(F_last,F):
        dates.append(date.toordinal(date.fromisoformat(data3["date"][j]))-t_min)
    if("Colombia" in countries[i])or("Sao Tome and Principe" in countries[i]):
        plt.scatter(dates[:],data3[F_last:F]["total_cases"],label="{}".format(countries[i]))
    F_last=F
plt.xlabel('time(days)')
plt.ylabel('Infected cases')
plt.legend(loc=(1.05,0.25))
################################################################
plt.figure()
F=0
F_last=0
for i in range(len(f)-2):
    F+=f[i]
    dates=[]
    for j in range(F_last,F):
        dates.append(date.toordinal(date.fromisoformat(data3["date"][j]))-t_min)
    if("Colombia" in countries[i])or("Sao Tome and Principe" in countries[i]):
        plt.scatter(dates[-n_data:],data3[F-n_data:F]["total_cases"],label="{}".format(countries[i]))
    F_last=F
plt.xlabel('time(days)')
plt.ylabel('Infected cases')
plt.legend(loc=(1.05,0.25))
F=0
F_last=0
X=[]
Y=[]
Y22=[]
for i in range(len(f)-2):
    F+=f[i]
    dates=[]
    for j in range(F_last,F):
        dates.append(date.toordinal(date.fromisoformat(data3["date"][j]))-t_min)
    if(f[i]>=21):
        X.append(dates[-n_data:])
        Y.append(np.array(data3[F-n_data:F]["total_cases"]))
        Y22.append(Y2[F-n_data:F])
    plt.plot(dates,data3[F_last:F]["total_cases"])#,label="{}".format(countries[i]))
    if("Colombia" in countries[i])or("United States"== countries[i])or("Peru" in countries[i]):
        plt.scatter(dates,data3[F_last:F]["total_cases"],label="{}".format(countries[i]))
        print(np.shape(data3[F-n_data:F]["total_cases"]))
    F_last=F
plt.xlabel('time(days)')
plt.ylabel('Infected cases')
plt.legend(loc=(1.05,0.25))
print(np.shape(X),np.shape(Y))
print(np.shape(Y[0][:]))
print(np.shape(Y22[0][:]))
# Vamos a hacer un split training test
scaler = StandardScaler()
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.5)
x_train = scaler.fit_transform(x_train).T
x_test = scaler.transform(x_test).T
#y_train = scaler.fit_transform(y_train.reshape(-1, 1))
#y_train=np.array(y_train).T
#y_test=np.array(y_test).T
#y_test = scaler.transform(y_test.reshape(-1, 1))
#x_train=x_train.T
print(np.shape(x_train))
print(np.shape(x_test))
print(np.shape(y_train[0][:]))
# x_train.mean(axis=0)
# x_train.std(axis=0)
#proba_test  = clf.predict_proba(np.array(X[0][:]).reshape(-1, 1))
#prec, rec, th = sklearn.metrics.precision_recall_curve(np.array(Y[0][:]).reshape(-1, 1), proba_test[:,1], pos_label=1)
proba_test_tot=[]
prec_tot=[]
rec_tot=[]
F1_tot=[]
for Ci in np.arange(1e3,1e5,1e3):
    clf = linear_model.LogisticRegression(C=Ci)
    # clf.fit(np.array(X[0][:]).reshape(-1, 1), np.array(Y[0][:]).reshape(-1, 1))
    clf.fit(x_train, y_train[0][:])
    proba_test  = clf.predict_proba(x_test)#.reshape(-1, 1))
    proba_test_tot.append(proba_test)
    prec, rec, th = sklearn.metrics.precision_recall_curve(y_test[0][:], proba_test[:,1], pos_label=1)
    prec_tot.append(prec)
    rec_tot.append(rec)
    F1= 2.0*prec[:-1]*rec[:-1]/(prec[:-1]+rec[:-1] +1E-10)
    F1_tot.append(F)
    print(proba_test[:,1])
    print(prec,"\t",rec,"\t",th)

F1= 2.0*prec[:-1]*rec[:-1]/(prec[:-1]+rec[:-1] +1E-10)
print(F1)
cnt=0
for i in range(len(data2["date"][:])):
    if("2020-04-28" in data2["date"][i]):
        print(data2["entity"][i].format())#data1.iloc[[i]]) #
        cnt+=1
print(cnt)

