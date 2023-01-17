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
data=pd.read_csv("/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")
data.head(10)
data.quality.value_counts()
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
sns.countplot(data.quality)
plt.figure(figsize=(10,10))
sns.heatmap(data.corr(),annot=True,cmap="rainbow")
sns.violinplot(x="quality",y='volatile acidity',data=data)
sns.boxplot(x="fixed acidity",data=data,color="y")
sns.scatterplot(x="pH",y='fixed acidity',data=data,hue="quality")
sns.swarmplot(x="quality",y='volatile acidity',data=data)
sns.pairplot(data,hue="quality")
sns.distplot(data.pH,color="r")
x=data.drop('quality',axis=1)
y=data['quality']
from imblearn.over_sampling import SMOTE
sm=SMOTE()
x,y=sm.fit_sample(x,y)
from sklearn.model_selection import train_test_split
xr,xt,yr,yt=train_test_split(x,y,test_size=0.1,random_state=42)
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
model=XGBClassifier(n_estimators=500)
model.fit(x,y)
yp=model.predict(xt)
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
print(accuracy_score(yt,yp))
print(classification_report(yt,yp))
sns.heatmap(confusion_matrix(yt,yp),annot=True,cmap="Purples")