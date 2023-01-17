# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')
data.head()
data.info()
data=data.drop(columns='Unnamed: 32')
data['diagnosis'].unique()
X=data.iloc[:,2:]
Y=data.iloc[:,1]
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X=scaler.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,Y,test_size=0.2, random_state=1)
np.sqrt(X_train.shape[0])
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
accuracy=[]
from sklearn.neighbors import KNeighborsClassifier
for i in range(1,50):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    accuracy.append(accuracy_score(y_test,knn.predict(X_test)))
plt.plot(range(1,50),accuracy)
accuracy.index(max(accuracy))
acc=pd.DataFrame(accuracy)
acc[acc[0]==max(accuracy)]
knn=KNeighborsClassifier(n_neighbors=25)
knn.fit(X_train ,y_train)
accuracy_score(y_test,knn.predict(X_test))
print(accuracy_score(Y,knn.predict(X)))
print('Difference:',accuracy_score(y_test,knn.predict(X_test))-accuracy_score(Y,knn.predict(X)))
