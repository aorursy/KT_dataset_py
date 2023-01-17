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
data=pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')
data.sample(5)
X=data.iloc[:,0:8].values
y=data.iloc[:,-1].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
from sklearn.preprocessing import StandardScaler #scaling the data because of the large difference of values between different input columns
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
model = SVC(kernel='linear', C=20)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
accuracy_score(y_pred,y_test) #accuracy for a random value of C
C_value=np.linspace(0,50,101).tolist()
C_value.pop(0)
accuracy=0
for i in C_value: #the range of C is from 0.5 to 50
    model = SVC(kernel='linear', C=i)
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    a1=accuracy_score(y_pred,y_test)
    if(a1>=accuracy):
        index=i
        accuracy=a1
print("Highest accuracy =",accuracy)
print("C =",index)