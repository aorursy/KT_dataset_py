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
data = pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')
data.head()
data.info()
data.columns
data.drop(['id','Unnamed: 32'],axis=1,inplace=True)
data.diagnosis.value_counts()
d={"diagnosis":{'B':0.0,'M':1.0}}
data.replace(d,inplace=True)
data.head()
corr_matrix=data.corr()

corr_matrix["diagnosis"].sort_values(ascending=False)
data.drop(["compactness_se","concavity_se","fractal_dimension_se","symmetry_se","texture_se","fractal_dimension_mean","smoothness_se"],axis=1,inplace=True)

labels=data["diagnosis"].values
data.drop(["diagnosis"],axis=1,inplace=True)


features=data.values
features
from sklearn import preprocessing
scaler=preprocessing.StandardScaler()
features_scaled=scaler.fit_transform(features)
features_scaled
from sklearn.model_selection import train_test_split
import numpy
numpy.random.seed(1234)
(x_train,x_test,y_train,y_test) = train_test_split(features_scaled,labels, train_size=0.75, random_state=42)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
clf=DecisionTreeClassifier(random_state=1)
scores=cross_val_score(clf,features_scaled,labels,cv=8)
scores
scores.mean()
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=10, random_state=42)
cv_scores = cross_val_score(clf, features_scaled,labels, cv=7)

cv_scores.mean()