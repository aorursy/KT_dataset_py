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
data=pd.read_csv('../input/drug-classification/drug200.csv')
data
from sklearn.preprocessing import StandardScaler,LabelEncoder
sc=StandardScaler()
l1=LabelEncoder()
l2=LabelEncoder()
l3=LabelEncoder()
l4=LabelEncoder()
data['Drug']=l1.fit_transform(data['Drug'])
data['Sex']=l2.fit_transform(data['Sex'])
data['Cholesterol']=l2.fit_transform(data['Cholesterol'])
data['BP']=l2.fit_transform(data['BP'])
data
data.corr()
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(data.drop('Drug',axis=1),data['Drug'])
model1=RandomForestClassifier()
model1.fit(X_train,Y_train)
print(model1.score(X_train,Y_train))
yp=model1.predict(X_test)
print(model1.score(X_test,Y_test))
df=[]
df=pd.DataFrame(df)
df['Actual']=Y_test.values
df['Predicted']=yp
df
