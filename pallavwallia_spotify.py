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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import tree
import imageio,io
import graphviz
df=pd.read_csv("../input/spotifyclassification/data.csv")
df.head()
df.describe()
df.shape
df.info()
type(df)
df.isnull().sum()
pos_tempo            = df[df['target']== 1]['tempo']
neg_tempo            = df[df['target']== 0]['tempo']
fig = plt.figure(figsize=(12,8))
plt.title('Song Tempo Like/Dislike distribution')
pos_tempo.hist(alpha=0.7,bins=30,label='positive')
neg_tempo.hist(alpha=0.7,bins=30,label='negative')
plt.legend()
plt.show()
dtc= DecisionTreeClassifier(min_samples_split=100)
dtc
X=df.drop(['Unnamed: 0','target','song_title','artist'],axis=1)
X.columns
features=['acousticness', 'danceability', 'duration_ms', 'energy',
       'instrumentalness', 'key', 'liveness', 'loudness', 'mode',
       'speechiness', 'tempo', 'time_signature', 'valence']

Y=df.target

Y=np.ravel(Y)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y)

X_train.shape

dtc.fit(X_train,Y_train)
dtc.score(X_train,Y_train)
y_pred=dtc.predict(X_test)
pd.DataFrame({'Actual':Y_test,'Predicted':y_pred})
score=accuracy_score(Y_test,y_pred)*100
score
