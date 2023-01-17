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
#Import the libraries

import numpy as np # linear algebra 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn

#Import and visualize the dataset

df=pd.read_csv("../input/mushroom-classification/mushrooms.csv")
df.head()
df.tail()
df.describe()
df.info()
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df = df.apply(label_encoder.fit_transform)
df
#Set the class as the response variable

y=df[['class']]

#Set the remaining variables in the dataframe as features

x=df.drop(columns=['class','veil-type','veil-color'])

X = (x - np.min(x))/(np.max(x)-np.min(x)).values
X.info
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

mushroom_model = LogisticRegression()
scores=cross_val_score(mushroom_model, X, y.values.ravel(),cv=10)
scores.mean()
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn, X, y.values.ravel(), cv=10, scoring='accuracy')
k_range = list(range(1, 31))
k_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y.values.ravel(), cv=10, scoring='accuracy')
    k_scores.append(scores.mean())
    
import matplotlib.pyplot as plt
%matplotlib inline

plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
knn = KNeighborsClassifier(n_neighbors=1)
print(cross_val_score(knn, X, y.values.ravel(), cv=10, scoring='accuracy').mean())
from sklearn.decomposition import PCA

pca=PCA()  
pca.n_components=20  
pca_data=pca.fit_transform(X)
percentage_var_explained = pca.explained_variance_ratio_;  
cum_var_explained=np.cumsum(percentage_var_explained)
 
plt.figure(1,figsize=(6,4))
plt.clf()  
plt.plot(cum_var_explained,linewidth=2)  
plt.axis('tight')  
plt.grid() 
plt.xlabel('n_components') 
plt.ylabel('Cumulative_Variance_explained')  
plt.show()
i_range=[0.7,0.8,0.9,0.95]
scores1=[]

for i in i_range:
  pca=PCA(i) 
  pca.fit(X) 
  X1=pca.transform(X) 
  mushroom_model1 = LogisticRegression()
  scores1=cross_val_score(mushroom_model1, X1, y.values.ravel(),cv=10)
  print(scores1.mean())
df=df.drop(columns=['veil-type','veil-color'])
plt.figure(figsize=(15, 10))
sn.heatmap(df.corr(), annot=True)
plt.show()