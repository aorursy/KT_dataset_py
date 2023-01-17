# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

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
data=pd.read_csv('/kaggle/input/league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv')
data.head()
data.info()
sns.heatmap(data.isnull(),cmap='Blues')
result=pd.DataFrame({'BlueWin':data['blueWins']})

result.head()
data.drop(['gameId'],inplace=True,axis=1)
data.head(1)
data.head(1)
blue_features=[]

red_features=[]

for col in list(data):

    if(col[0]=='r'):

        red_features.append(col)

    if(col[0]=='b'):

        blue_features.append(col)
blue_features
blue=data[blue_features]

red_features.append("blueWins")

red=data[red_features]
blue.head()
red.head()
g=sns.PairGrid(data=red,hue='blueWins',palette='Set1')

g.map_diag(plt.hist)

g.map_offdiag(plt.scatter)

g.add_legend();
g=sns.PairGrid(data=blue,hue='blueWins',palette='Set1')

g.map_diag(plt.hist)

g.map_offdiag(plt.scatter)

g.add_legend();
Red_win=len(data[data['blueWins']==0])

Blue_win=len(data['blueWins'])-len(data[data['blueWins']==1])

print(Blue_win)

print(Red_win)
g = sns.countplot(x=data['blueWins'])
plt.figure(figsize=(16,5))

sns.heatmap(red.corr(),annot=True,cmap='Reds')
red.drop(['redWardsPlaced','redWardsDestroyed','redFirstBlood','redHeralds','redTowersDestroyed','redTotalJungleMinionsKilled','blueWins'],axis=1,inplace=True)
red.head(1)
plt.figure(figsize=(18,5))

sns.heatmap(blue.corr(),annot=True,cmap='Blues')
blue.drop(['blueTotalJungleMinionsKilled','blueWardsPlaced','blueWardsDestroyed','blueFirstBlood','blueHeralds','blueTowersDestroyed'],axis=1,inplace=True)
blue.head(1)
final_data=pd.concat([red,blue],axis=1)
final_data.head(2)
from sklearn.metrics import classification_report,confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

from sklearn import preprocessing 

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
x=data.drop('blueWins',axis=1)

y=data['blueWins']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=9)
final_LR=LogisticRegression()
final_LR.fit(x_train,y_train)
result_LR=final_LR.predict(x_test)

print(accuracy_score(result_LR,y_test))
x_mod=final_data.drop('blueWins',axis=1)

y_mod=final_data['blueWins']
x_mod_train,x_mod_test,y_mod_train,y_mod_test=train_test_split(x_mod,y_mod,test_size=0.3,random_state=9)
mod_LR=LogisticRegression()
mod_LR.fit(x_mod_train,y_mod_train)
Mod_result=mod_LR.predict(x_mod_test)
accuracy_score(Mod_result,y_mod_test)
x=data.drop('blueWins',axis=1)

y=data['blueWins']
x=preprocessing.StandardScaler().fit_transform(x)
pca=PCA(n_components=2)
components=pca.fit_transform(x)
plt.figure(figsize=(10,8))

plt.scatter(components[:,0],components[:,1],c=y,cmap='plasma')

plt.xlabel('First Principal Comp')

plt.ylabel('Second Principal COmp')
x=final_data.drop('blueWins',axis=1)

y=data['blueWins']
x=preprocessing.StandardScaler().fit_transform(x)
pca=PCA(n_components=3)
components=pca.fit(x)
transfrom=components.transform(x)
plt.figure(figsize=(10,8))

plt.scatter(transfrom[:,0],transfrom[:,1],c=y,cmap='plasma')

plt.xlabel('First Principal Comp')

plt.ylabel('Second Principal COmp')
components.components_
final_data.drop('blueWins',axis=1,inplace=True)
q=pd.DataFrame(components.components_,columns=final_data.columns)

q
plt.figure(figsize=(12,6))

sns.heatmap(q,cmap='plasma')