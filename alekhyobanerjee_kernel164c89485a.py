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
from sklearn.preprocessing import LabelEncoder,LabelEncoder,StandardScaler

from sklearn.model_selection import train_test_split,GridSearchCV

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
data=pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')
data
data=data.iloc[:,:-1]
data.isnull().any()
sns.countplot(data['diagnosis'])

plt.show()
X=data.iloc[:,2:].values

Y=data.iloc[:,1].values
le=LabelEncoder()

Y=le.fit_transform(Y)
list(le.inverse_transform([0,1]))
clf=DecisionTreeClassifier()
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
clf.fit(X_train,Y_train)

Y_pred=clf.predict(X_test)

accuracy_clf_without_tuning=accuracy_score(Y_test,Y_pred)
param_dist={

    "max_depth":[1,2,3,4,5,None],

    "criterion":["gini","entropy"],

    "min_samples_leaf":[1,2,3,4,5],

    "min_samples_split":[1,2,3,4,5],

    "splitter":["random","best"]

}
grid=GridSearchCV(clf,param_grid=param_dist,cv=10,n_jobs=-1)

grid.fit(X_train,Y_train)

accuracy_clf_with_tuning=grid.best_score_
print("Accuracy of Decision Tree classifier with default hyperparameters is ",round(accuracy_clf_without_tuning*100,2))

print("Accuracy of Decison Tree classifier with hyperparameters as {} is {}".format(grid.best_params_,round(accuracy_clf_with_tuning*100,2)))
#Scalarisation of Data

scaler=StandardScaler()

X=scaler.fit_transform(X)

k=np.sqrt(X_train.shape[0])
k
knn=KNeighborsClassifier(n_neighbors=21)

knn.fit(X_train,Y_train)

Y_pred=knn.predict(X_test)    



accuracy_knn_without_tuning=accuracy_score(Y_test,Y_pred)
#Trial and Error method to find the best K for better accuracy



accuracy=[]

for i in range(1,31):

    knn=KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,Y_train)

    Y_pred=knn.predict(X_test)    

    accuracy.append(accuracy_score(Y_test,Y_pred))

    

accuracy=np.array(accuracy)
print("The Maximum Accuracy can be achieved is {} where K is {}".format(round(np.max(accuracy)*100,2),np.argmax(accuracy)+1))