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
import matplotlib.pyplot as plt
import numpy
data=pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')
data
data['diagnosis']=data['diagnosis'].replace({'M':0,'B':1})
data=data.drop(['Unnamed: 32','id'],axis=1)
data
import seaborn as sns
sns.pairplot(data,hue='diagnosis')
data.info()
plt.scatter(x=data['radius_mean'],y=data['texture_mean'],c=data['diagnosis'])
plt.xlabel('Radius Mean')
plt.ylabel('Texture Mean')
plt.scatter(x=data['radius_se'],y=data['texture_se'],c=data['diagnosis'])
plt.xlabel('Radius ')
plt.ylabel('Texture ')
plt.scatter(x=data['radius_worst'],y=data['texture_worst'],c=data['diagnosis'])
plt.xlabel('Radius Worst')
plt.ylabel('Texture Worst')
plt.scatter(x=data['radius_mean'],y=data['fractal_dimension_worst'],c=data['diagnosis'])
plt.xlabel('Radius Mean')
plt.ylabel('fractal_dimension_worst')
plt.scatter(x=data['concavity_worst'],y=data['compactness_worst'],c=data['diagnosis'])
plt.xlabel('Concavity_worst')
plt.ylabel('Compactness_worst')
plt.scatter(x=data['concavity_mean'],y=data['compactness_mean'],c=data['diagnosis'])
plt.xlabel('Concavity_Mean')
plt.ylabel('Compactness_Mean')
X=data.drop('diagnosis',axis=1)
Y=data['diagnosis']
X
X.corr()
data.corr()
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y)
X_train
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=200)
model.fit(X_train,Y_train)
model.score(X_train,Y_train)
model.score(X_test,Y_test)
y_pre=model.predict(X_test)
from sklearn.metrics import f1_score,classification_report
f1_score(y_pre,Y_test)
