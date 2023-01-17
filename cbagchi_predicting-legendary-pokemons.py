# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/Pokemon.csv")

data.head()
X=data[["Total","HP","Attack","Defense","Sp. Atk","Sp. Def","Speed"]]

y=data["Legendary"]



scaler=StandardScaler().fit(X)   # Z-scaling the data

X=scaler.transform(X)

y=y.astype(int)
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.33)

pd.DataFrame(X_train).head()
from sklearn.neighbors import KNeighborsClassifier

clf=KNeighborsClassifier(n_neighbors=5)

clf=clf.fit(X_train,y_train)

pred_knn=clf.predict(X_test)

accuracy=sum(pred_knn==y_test)/pred_knn.shape[0]*100

print("Accuracy=",accuracy)
from sklearn.linear_model import LogisticRegression

logReg=LogisticRegression()

logReg=logReg.fit(X_train,y_train)

pred_LR=logReg.predict(X_test)

accuracy=sum(pred_LR==y_test)/pred_knn.shape[0]*100

print("Accuracy=",accuracy)
from sklearn import svm

clf=svm.SVC()

clf=clf.fit(X_train,y_train)

pred_svm=clf.predict(X_test)

accuracy=sum(pred_svm==y_test)/pred_knn.shape[0]*100

print("Accuracy=",accuracy)
from sklearn.ensemble import RandomForestClassifier

clf=RandomForestClassifier(n_estimators=100,max_depth=5)

clf=clf.fit(X_train,y_train)

pred_rf=clf.predict(X_test)

accuracy=sum(pred_rf==y_test)/pred_knn.shape[0]*100

print("Accuracy=",accuracy)