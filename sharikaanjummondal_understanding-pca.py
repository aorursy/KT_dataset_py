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
df=pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_train.csv')
df.shape
df.head()
X=df.iloc[:,1:].values
y=df.iloc[:,0].values
import matplotlib.pyplot as plt
a=X[0].reshape(28,28)
plt.imshow(a)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
from sklearn.neighbors import KNeighborsClassifier
clf=KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
from sklearn.metrics import accuracy_score

accuracy_score(y_test,y_pred)
#Feature Scaling



from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train=sc.fit_transform(X_train)

X_test = sc.transform(X_test)
#Apply PCA



from sklearn.decomposition import PCA

pca=PCA(n_components=200)

X_train_new = pca.fit_transform(X_train)

X_test_new=pca.transform(X_test)
print(X_train.shape)

print(X_train_new.shape)
pca.explained_variance_ratio_
clf.fit(X_train_new,y_train)
y_pred_new = clf.predict(X_test_new)
accuracy_score(y_test,y_pred_new)