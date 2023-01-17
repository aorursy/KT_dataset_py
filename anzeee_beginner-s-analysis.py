

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')

df.head()
df.columns
df.shape
df.info()
df.drop(['Unnamed: 32','id'],axis=1,inplace=True)
df.head()
y=df.diagnosis

x=df.drop('diagnosis',axis=1)
y.value_counts()
ax = sns.countplot(y,label="Count")
scaler=StandardScaler()

x_scaled=scaler.fit_transform(x)

x_scaled
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25,random_state=30)
from sklearn.linear_model import LogisticRegression

reg = LogisticRegression(max_iter=10000)
reg.fit(x_train,y_train)



reg.score(x_test,y_test)