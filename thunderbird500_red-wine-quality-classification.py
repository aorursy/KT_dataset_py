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
df = pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
df.head(10)
df.describe()
df.info()
df.isnull().sum()
df['quality'].unique()
df.corr()
plt.figure(figsize= (16,10))
sns.heatmap(df.corr(),cmap = 'Dark2',annot = True,linewidths=1.0,linecolor='black')
df['good_quality'] = [1 if x>=7 else 0 for x in df['quality']]
from sklearn.model_selection import train_test_split
X = df.drop(['quality','good_quality'],axis=1)
y = df['good_quality']
df['good_quality'].value_counts()
from sklearn.preprocessing import StandardScaler
X_features = X
X = StandardScaler().fit_transform(X)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
from sklearn.ensemble import RandomForestClassifier
model1 = RandomForestClassifier()
model1.fit(X_train,y_train)
y_pred1 = model1.predict(X_test)
y_test.shape,y_pred1.shape
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred1))