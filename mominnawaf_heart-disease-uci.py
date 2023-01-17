# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier 
df=pd.read_csv("../input/heart-disease-uci/heart.csv")
df.info()
df.describe()
import seaborn as sns
corrmat=df.corr()

corrmat
top_cor_features=corrmat.index

top_cor_features
plt.figure(figsize=(20,20))


g=sns.heatmap(df[top_cor_features].corr(),annot=True,cmap="RdYlGn")
df.hist()
sns.set_style('whitegrid')

sns.countplot(x='target',data=df,palette="RdBu_r")
dataset=pd.get_dummies(df,columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
StandardScaler = StandardScaler()
colums_to_scale= ['age','trestbps','chol','thalach','oldpeak']
dataset[colums_to_scale]=StandardScaler.fit_transform(dataset[colums_to_scale])
dataset
y=dataset["target"]

x=dataset.drop(["target"],axis=1)
from sklearn.model_selection import cross_val_score

knn=[]

for k in range(1,21):

    knn_classifier = KNeighborsClassifier (n_neighbors = k)

    score=cross_val_score(knn_classifier,x,y,cv=20)

    knn.append(score.mean())
plt.plot([k for k in range (1,21)],knn,color="red")

for i in range (1,21):

    plt.text(i, knn[i-1], (i, knn[i-1]))



    
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=10)
score=cross_val_score(rfc,x,y,cv=151)
score
score.mean()