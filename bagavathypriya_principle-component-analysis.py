# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.datasets import load_breast_cancer
cancer=load_breast_cancer()
type(cancer)
cancer.keys()
print(cancer['DESCR'])
df=pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
df.head()
df.info()
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(df)
scaled_df=scaler.transform(df)
from sklearn.decomposition import PCA 
pca=PCA(n_components=2)
pca.fit(scaled_df)
pca_df=pca.transform(scaled_df)
pca_df.shape
pca_df
plt.figure(figsize=(8,6))
plt.scatter(pca_df[:,0],pca_df[:,1],c=cancer['target'])
plt.xlabel('First principal component')
plt.ylabel('Second Principal component')
plt.legend()
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(pca_df,cancer['target'],test_size=0.2)
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(xtrain,ytrain)
pred=knn.predict(xtest)
from sklearn.metrics import classification_report,confusion_matrix
con=confusion_matrix(ytest,pred)
print(con)
rep=classification_report(ytest,pred)
print(rep)
from sklearn.model_selection import cross_val_score
err=[]
for i in range(1,40):
    knn=KNeighborsClassifier(n_neighbors=i)
    score=cross_val_score(knn,pca_df,cancer['target'],cv=10)
    err.append(1-score.mean())
plt.figure(figsize=(10,6))
plt.plot(range(1,40),err,linestyle='dashed',marker='o',markersize=10,markerfacecolor='red')
plt.xlabel('K-neighbor')
plt.ylabel('Error rate')
knn = KNeighborsClassifier(n_neighbors=21)

knn.fit(xtrain,ytrain)
pred = knn.predict(xtest)

print('WITH K=21')
print('\n')
print(confusion_matrix(ytest,pred))
print('\n')
print(classification_report(ytest,pred))
