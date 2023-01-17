# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd# data processing, CSV file I/O (e.g. pd.read_csv)
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
dataset = pd.read_csv('/kaggle/input/fish-market/Fish.csv')
dataset.head()
dataset.Species.value_counts(dropna=False).head()
dataset.info()
sns.pairplot(dataset,x_vars = ['Weight','Length1','Length2','Length3','Height','Width'],y_vars = 'Species',height=5,aspect=0.7)
sns.heatmap(dataset.iloc[:,1:].corr(),yticklabels=True,cbar=True,cmap='cividis')
f = dataset.iloc[:,0].unique().tolist()
print(f)
for fish in f:
    dataset.loc[dataset.iloc[:,0]==fish,['Species']] = f.index(fish)
dataset.iloc[:,0]=dataset.iloc[:,0].astype(int)
X = dataset.iloc[:,1:].values
y = dataset['Species'].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(C=15,max_iter=200,random_state=0,penalty = 'l2')
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
classifier.score(X_test,y_test)
from sklearn.metrics import confusion_matrix,r2_score
cm = confusion_matrix(y_test,y_pred)
print(cm)
r2_score(y_test,y_pred)

c = np.array(input().split()).astype(float)

d = c.reshape(1,-1)
v = classifier.predict(d)
f[v[0]]