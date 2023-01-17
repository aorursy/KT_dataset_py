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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
data=pd.read_csv("/kaggle/input/iris/Iris.csv")
data
data.shape

data['Species'].replace({'Iris-setosa':0,'Iris-virginica':1,'Iris-versicolor':2},inplace=True)
X=data[['PetalLengthCm','SepalLengthCm']]#Act as a feature variable
print(X)
Y=data['Species']#Act as a target variable
print(Y)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=9)
X_train.shape
Y_train.shape
#Training the model using Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier()
clf.fit(X_train,Y_train)
y_pred=clf.predict(X_test)
print("Predicted value is:",y_pred)
from sklearn.metrics import accuracy_score
acc=accuracy_score(Y_test,y_pred)
print()
print("Accuracy of the model is:",acc)


a=np.arange(start=X_train.iloc[:,0].min()-1,stop=X_train.iloc[:,0].max()+1,step=0.01)
b=np.arange(start=X_train.iloc[:,1].min()-1,stop=X_train.iloc[:,1].max()+1,step=0.01)

XX,YY=np.meshgrid(a,b)

print(XX.shape[0])
print(XX.shape[1])
print(YY.shape[0])
print(YY.shape[1])
np.array([XX.ravel(),YY.ravel()]).T.shape
input_array=np.array([XX.ravel(),YY.ravel()]).T

labels=clf.predict(input_array)

labels.shape


plt.contourf(XX,YY,labels.reshape(XX.shape),alpha=0.5)

plt.scatter(X_train.iloc[:,0],X_train.iloc[:,1],c=Y_train)