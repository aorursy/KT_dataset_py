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
df=pd.read_csv("/kaggle/input/breast-cancer-wisconsin-data/data.csv")
df.head()
print(df.info())
df.drop(columns=["Unnamed: 32","id"],inplace=True)#removing unnecessary features
target=df[["diagnosis"]].values
x=df.iloc[:,1:].values
from sklearn.preprocessing import StandardScaler

ss=StandardScaler()

ss.fit(x)

x_std=ss.transform(x)
from sklearn.decomposition import PCA
pc=PCA(n_components=6)

pc.fit(x_std)

x_pca=pc.transform(x_std)
import matplotlib.pyplot as plt

import seaborn as sns

sns.heatmap(pc.components_,cmap='plasma')
from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(x_pca,target)

from xgboost import XGBClassifier

xclf=XGBClassifier()

xclf.fit(xtrain,ytrain)

from sklearn.metrics import accuracy_score,recall_score,confusion_matrix
accuracy_score(xclf.predict(xtest),ytest)
confusion_matrix(xclf.predict(xtest),ytest)