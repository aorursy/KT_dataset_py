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
data=pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")
data.head()
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
style.use("dark_background")
plt.figure(figsize=(10,10))
sns.heatmap(data.corr(),annot=True,cmap='rainbow')
sns.countplot(data.target)
sns.swarmplot(x="target",y="thalach",data=data,color='r')
sns.violinplot(x="target",y="thalach",data=data)
plt.figure(figsize=(15,10))
sns.scatterplot(data['trestbps'],data['thalach'],color='r')
sns.lineplot(data['chol'],data['thalach'])
x=data.drop("target",axis=1)
y=data["target"]
from imblearn.over_sampling import SMOTE
sm=SMOTE()
x,y=sm.fit_sample(x,y)
from sklearn.model_selection import train_test_split
xr,xt,yr,yt=train_test_split(x,y,test_size=0.1)
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
model=XGBClassifier(n_estimators=1000)
print(model)
model.fit(x,y)
yp=model.predict(xt)
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
print(accuracy_score(yt,yp))
print(classification_report(yt,yp))
sns.heatmap(confusion_matrix(yt,yp),annot=True,cmap='rainbow')