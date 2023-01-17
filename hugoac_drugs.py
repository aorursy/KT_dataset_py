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
df = pd.read_csv('/kaggle/input/drug-classification/drug200.csv')
df.sample(5)
df.info()
v_cont = df.describe().columns.tolist()
v_disc = [v for v in df.columns.drop('Drug') if v not in v_cont]
y = df['Drug']
df= df[v_cont].join(pd.get_dummies(df[v_disc]))
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
Rf = RandomForestClassifier()
X = df.copy()
X_t,X_v,y_t,y_v = train_test_split(X,y,train_size=0.8)
print('X_t = '+str(X_t.shape))

print('y_t = '+str(y_t.shape))

print('X_v = '+str(X_v.shape))

print('y_v = '+str(y_v.shape))
Rf.fit(X_t,y_t)
from sklearn.metrics import confusion_matrix, accuracy_score
confusion_matrix(y_v,Rf.predict(X_v))
accuracy_score(y_v,Rf.predict(X_v))