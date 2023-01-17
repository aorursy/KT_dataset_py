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
df=pd.read_csv('/kaggle/input/bank-note-authentication-uci-data/BankNote_Authentication.csv')
df.head()
df.describe()
X=df.drop(labels=['class'],axis=1)
y=df['class']
print(X.shape, y.shape)
import seaborn as sns
import matplotlib.pyplot as plt
sns.pairplot(df,hue='class')
plt.show()
'''from sklearn.decomposition import PCA
pca=PCA(n_components=2)
X=pca.fit_transform(X)'''
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
X=ss.fit_transform(X)


plt.scatter(X[:,0],X[:,1],c=y)
plt.show()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2, random_state=23)
print(X_train.shape, X_test.shape)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(X_train,y_train)
from sklearn.metrics import accuracy_score
y_pred=lr.predict(X_test)
score=accuracy_score(y_test,y_pred)
score
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(X_train,y_train)
from sklearn.metrics import accuracy_score
y_pred=knn.predict(X_test)
score=accuracy_score(y_test,y_pred)
score
import pickle
pickle_out = open("knnmodel.pkl","wb")
pickle.dump(knn, pickle_out)
pickle_out.close()







