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
data=pd.read_csv("/kaggle/input/star-dataset/6 class csv.csv")
data.head()
import matplotlib.pyplot as plt
import seaborn as sns
import math as m
data.describe()
#Strange results for Luminosity and Radius: The quartiles have a sudden change in the order of magnitude
data.info()
sns.pairplot(data)
#checking out the relation of Log(T) with all variables
temp_log=data["Temperature (K)"]
temp_log=temp_log.apply(m.log10)
data2=data
data2["Temperature (K)"]=temp_log
sns.pairplot(data2)
#Rescaling Luminosity and Radius
rad_log=data["Radius(R/Ro)"]
lum_log=data["Luminosity(L/Lo)"]
rad_log=rad_log.apply(m.log10)
lum_log=lum_log.apply(m.log10)
data4=data
data4["Radius(R/Ro)"]=rad_log
data4["Luminosity(L/Lo)"]=lum_log
sns.pairplot(data4)

data4["log(L/Lo)"]=data4["Luminosity(L/Lo)"]
data4["log(R/Ro)"]=data4["Radius(R/Ro)"]
data4.drop(["Luminosity(L/Lo)","Radius(R/Ro)"], axis=1, inplace=True)
data4.describe()
#Luminosity and Radius look better know
#from now on the datax dataset is refering to the data that has model x for spectral class
data1=data4.drop(["Star color","Spectral Class"], axis=1)
data2=data4.drop("Star color", axis=1)
#Checking all categories in both features
data4["Spectral Class"].unique()
data4["Star color"].unique()
#For data2 I'll use the following dictionary
sp_class={"O":0,"B":1,"A":2,"F":3,"G":4,"K":5,"M":6}
data2["Spectral Class"]=data2["Spectral Class"].map(sp_class)
#making dummies for light color and for spectral class
dumm_light=pd.get_dummies(data4["Star color"], drop_first=True)
dumm_class=pd.get_dummies(data4["Spectral Class"], drop_first=True)
data1=data1.join(dumm_light)
data1=data1.join(dumm_class)
data2=data2.join(dumm_light)
#checking datasets
data1.head()
data2.head()
#sklearn imports
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
#first organize the test train splits
X1=data1.drop("Star type", axis=1)
y1=data1["Star type"]
X2=data2.drop("Star type", axis=1)
y2=data2["Star type"]
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.3)
lr1=LogisticRegression(max_iter=10000)
lr1.fit(X1_train,y1_train)
result_lr1=lr1.predict(X1_test)
print(confusion_matrix(y1_test,result_lr1))
print("\n")
print(classification_report(y1_test,result_lr1))
lr2=LogisticRegression(max_iter=10000)
lr2.fit(X2_train,y2_train)
result_lr2=lr2.predict(X2_test)
print(confusion_matrix(y2_test,result_lr2))
print("\n")
print(classification_report(y2_test,result_lr2))
lda1=LinearDiscriminantAnalysis()
lda1.fit_transform(X1_train,y1_train)
result_lda1=lda1.predict(X1_test)
print(confusion_matrix(y1_test,result_lda1))
print("\n")
print(classification_report(y1_test,result_lda1))
lda2=LinearDiscriminantAnalysis()
lda2.fit_transform(X2_train,y2_train)
result_lda2=lda2.predict(X2_test)
print(confusion_matrix(y2_test,result_lda2))
print("\n")
print(classification_report(y2_test,result_lda2))
#Decision of K value
import numpy as np
error_rate1=[]
error_rate2=[]
for i in range(1,50):
    knn1=KNeighborsClassifier(n_neighbors=i)
    knn2=KNeighborsClassifier(n_neighbors=i)
    knn1.fit(X1_train,y1_train)
    knn2.fit(X2_train,y2_train)
    res_knn1_i=knn1.predict(X1_test)
    res_knn2_i=knn2.predict(X2_test)
    error_rate1.append(np.mean(res_knn1_i!=y1_test))
    error_rate2.append(np.mean(res_knn2_i!=y2_test))
n_values=range(1,50)
plt.fig_size=(12,20)
plt.plot(n_values,error_rate1)
plt.ylabel("Error rate")
plt.xlabel("K value")
neighbors1=error_rate1.index(min(error_rate1))+1
n_values=range(1,50)
plt.plot(n_values,error_rate2)
plt.ylabel("Error rate")
plt.xlabel("K value")
neighbors2=error_rate2.index(min(error_rate2))+1
knn1=KNeighborsClassifier(n_neighbors=neighbors1)
knn1.fit(X1_train,y1_train)
result_knn1=knn1.predict(X1_test)
print(confusion_matrix(y1_test,result_knn1))
print("\n")
print(classification_report(y1_test,result_knn1))
knn2=KNeighborsClassifier(n_neighbors=neighbors2)
knn2.fit(X2_train,y2_train)
result_knn2=knn2.predict(X2_test)
print(confusion_matrix(y2_test,result_knn2))
print("\n")
print(classification_report(y2_test,result_knn2))
rf1=RandomForestClassifier(n_estimators=300)
rf1.fit(X1_train,y1_train)
result_rf1=rf1.predict(X1_test)
print(confusion_matrix(y1_test,result_rf1))
print("\n")
print(classification_report(y1_test,result_rf1))
rf2=RandomForestClassifier(n_estimators=300)
rf2.fit(X2_train,y2_train)
result_rf2=rf2.predict(X2_test)
print(confusion_matrix(y2_test,result_rf2))
print("\n")
print(classification_report(y2_test,result_rf2))