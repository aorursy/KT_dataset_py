# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')

df.head()
df.info()
df.describe().T
sns.pairplot(df)
df = df.drop( columns = ['id','Unnamed: 32'])
df.corr()
x_data = df.drop('diagnosis',axis=1)

y_data = df['diagnosis']

y_mapped = df['diagnosis'].map({'B':0,

                                 'M':1})

x_train,x_test,y_train,y_test = train_test_split(x_data,y_mapped,test_size=0.2,random_state=42)
sc = StandardScaler()

x_train_scaled = sc.fit_transform(x_train)

x_test_scaled  = sc.transform(x_test)
pc = PCA(n_components=2)

x_train_compact = pc.fit_transform(x_train_scaled)

x_test_compact  = pc.transform(x_test_scaled)
lr = LogisticRegression()

model = lr.fit(x_train_compact,y_train)
y_predict = model.predict(x_test_compact)
model.score(x_test_compact,y_test)
print(confusion_matrix(y_test,y_predict))
print(classification_report(y_test,y_predict))