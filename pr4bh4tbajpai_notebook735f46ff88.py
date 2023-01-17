# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
#from mpl_toolkits.mplot3d import Axes3D
#from sklearn.preprocessing import StandardScaler
#import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df1=pd.read_csv('/kaggle/input/bitcoinheistransomwareaddressdataset/BitcoinHeistData.csv',delimiter=',')
df1.dataframeName='BitcoinHeistData.csv'
nRow,nCol=df1.shape
print(f'there are {nRow} and {nCol} Colums')
#df1=df1.drop("address",axis=1)
df1.head(10)
from sklearn import preprocessing 
label_encoder = preprocessing.LabelEncoder() 
df1['label']= label_encoder.fit_transform(df1['label']) 
df1['address']=label_encoder.fit_transform(df1['address'])
print(df1.head(20))
def f1():
    X=df1.drop('label',axis=1)
    y=df1['label']
    return X,y
f1()
from sklearn.model_selection import train_test_split
def f2():
    X,y=f1()
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
    return X_train,X_test,y_train,y_test
f2()
from sklearn.tree import DecisionTreeClassifier
def f4():
    X_train,X_test,y_train,y_test =f2()
    dtc=DecisionTreeClassifier(criterion='entropy',max_depth=4)
    dtc.fit(X_train,y_train)
    return dtc.score(X_test,y_test)
f4()

from sklearn.neighbors import KNeighborsClassifier
def f3():
    X_train,X_test,y_train,y_test =f2()
    knn=KNeighborsClassifier(n_neighbors=10)
    knn.fit(X_train,y_train)
    return knn.score(X_test,y_test)
f3()
from sklearn.ensemble import RandomForestClassifier
def f5():
    X_train,X_test,y_train,y_test=f2()
    rf=RandomForestClassifier(max_depth=3,random_state=0)
    rf=rf.fit(X_train,y_train)
    return rf.score(X_test,y_test)
f5()
# from sklearn.ensemble import GradientBoostingClassifier
# def f6():
#     X_train,X_test,y_train,y_test=f2()
#     gb=GradientBoostingClassifier(random_state=0)
#     gb=gb.fit(X_train,y_train)
#     return gb.score(X_test,y_test)
# f6()
from sklearn.ensemble import AdaBoostClassifier
def f7():
    X_train,X_test,y_train,y_test=f2()
    ab=AdaBoostClassifier(n_estimators=100,random_state=0)
    ab.fit(X_train,y_train)
    return ab.score(X_test,y_test)
f7()