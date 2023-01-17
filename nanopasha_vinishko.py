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
from sklearn.datasets import load_wine

wine_data = load_wine ()

df=pd.DataFrame (wine_data['data'],columns=wine_data['feature_names'])

df["Target"] = wine_data ["target"]
df.head ()

x=df.drop("Target",axis=1)

y=df["Target"]
from sklearn.neighbors import KNeighborsClassifier
clf=KNeighborsClassifier(n_neighbors=100)
clf.fit(x,y)
df.iloc[92]["Target"]
x.iloc[92]
clf.predict([x.iloc[92]])
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

x_scaled= scaler.fit_transform(x)
score=[]

for i in range (1,50):

    clf=KNeighborsClassifier(n_neighbors=i)

    cv = KFold(n_splits=5,shuffle=True, random_state=42)

    val=cross_val_score(clf,x_scaled,y,cv=cv)

    score.append(val.mean())
import matplotlib.pyplot as plt

%matplotlib inline
plt.plot(score)
max(score)
score.index(max(score))+1