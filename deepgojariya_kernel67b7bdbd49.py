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
df = pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
df.head()
df.isnull().sum()
df.corr()
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(20,20))
sns.heatmap(df.corr(), annot = True)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
df.shape
X=df.iloc[:,:-1]
y=df.iloc[:,11:]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)
reg1 = LinearRegression()
reg1.fit(X_train,y_train)
y_pred = reg1.predict(X_test)
y_pred=pd.DataFrame(y_pred,columns=['y_pred'])
y_pred
from sklearn.metrics import *
#reg1.score(y_pred)
