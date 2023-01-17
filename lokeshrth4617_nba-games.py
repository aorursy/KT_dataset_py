# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/nba-games-stats-from-2014-to-2018/nba.games.stats.csv')

data
data.info()
data['Date'] = pd.to_datetime(data['Date'])
data['year'] = data.Date.dt.year

data['year']
CLE = data[(data['Team']=='CLE')]
data = CLE.copy()
data.columns
data['WINorLOSS'] = data['WINorLOSS'].replace('W',1)

data['WINorLOSS'] = data['WINorLOSS'].replace('L',0)
data['WINorLOSS'] = data['WINorLOSS'].astype('int')
CLE['Home'] = CLE['Home'].replace('Home',1)

CLE['Home'] = CLE['Home'].replace('Away',0)

data.columns
CLE7 = data.drop(['Unnamed: 0','Date','Team','year'],axis = 1)
CLE7['Home'] = CLE7['Home'].replace('Home',1)

CLE7['Home'] = CLE7['Home'].replace('Away',0)

CLE7
dummies = pd.get_dummies(CLE['Opponent'])

dummies.head()
data1 = pd.concat([dummies,CLE7],axis =1)
data1 = data1.drop('Opponent',axis = 1)
X = data1.drop(['WINorLOSS'],axis = 1)

X
Y = data1['WINorLOSS']

Y = Y[:,np.newaxis]
print(Y.shape)

print(X.shape)
from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
y_test.shape
from sklearn.linear_model import LogisticRegression

Lr = LogisticRegression()

Lr.fit(X_train,y_train)
y_pred = Lr.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score

score = accuracy_score(y_test,y_pred)

score
from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score



cm = confusion_matrix(y_test, y_pred)

print(cm)

print(accuracy_score(y_test, y_pred))