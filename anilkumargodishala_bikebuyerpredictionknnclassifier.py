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
dataset=pd.read_csv('../input/bikebuyer/BikeBuyer.csv')
dataset

dataset.head(5)
dataset.isnull().any()
print(dataset["Gender"].unique())
print(dataset["Marital Status"].unique())
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
le
dataset["Marital Status"]=le.fit_transform(dataset["Marital Status"])
dataset["Marital Status"]
dataset["Gender"]=le.fit_transform(dataset["Gender"])
dataset["Gender"]
dataset
dataset["BikeBuyer"]=le.fit_transform(dataset["BikeBuyer"])
dataset["BikeBuyer"]
dataset
x=dataset.iloc[:,[1,2,3,4,11]].values
x
y=dataset.iloc[:,12].values
y
x.shape
y.shape
from sklearn.model_selection import train_test_split
x_train ,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
x_train

x_train.shape
x_test.shape
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train =sc.fit_transform(x_train)
x_test =sc.fit_transform(x_test)
x_train
x_test
from sklearn.neighbors import KNeighborsClassifier
knn =KNeighborsClassifier(n_neighbors=5,metric ='minkowski',p=2)
knn
knn.fit(x_train,y_train)
kpred=knn.predict(x_test)
kpred
y_test
from sklearn.metrics import accuracy_score
kaccuracy = accuracy_score(y_test,kpred)
kaccuracy
from sklearn.metrics import confusion_matrix
kcm=confusion_matrix(y_test,kpred)
kcm
import sklearn.metrics as metrics
kfpr,ktpr,rtreshhold = metrics.roc_curve(y_test,kpred)
kfpr
ktpr
rtreshhold
kroc_auc = metrics.auc(kfpr,ktpr)
kroc_auc
import matplotlib.pyplot as plt
plt.title("Receiver Operating Characteristics")
plt.plot(kfpr,ktpr,color="green",label="auc_roc curve value=%0.2f"%kroc_auc)
plt.legend()
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel("true positive rate")
plt.xlabel("false posoitive rate")
