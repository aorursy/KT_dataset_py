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
data=pd.read_csv('../input/titanic/train.csv')

datas=pd.read_csv('../input/titanic/test.csv')
data.isnull().sum()

datas.isnull().sum()
data=data.fillna(1)

datas=datas.fillna(1)

datas
data.isnull().sum().sum()

datas.isnull().sum().sum()
data=pd.get_dummies(data,columns=['Sex'])

datas=pd.get_dummies(datas,columns=['Sex'])
import seaborn as sns

import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))

cor=data.corr()

sns.heatmap(cor,annot=True,cmap=plt.cm.Reds)

plt.show()
target=abs(cor['Survived'])

required=target[target>0.5]

required
print(data[['Sex_male','Sex_female']].corr())
X=data.loc[:,['Sex_male','Sex_female']]

Y=data.loc[:,'Survived']

print(Y)
X_test=datas.loc[:,['Sex_male','Sex_female']]

Id=datas.loc[:,'PassengerId']

from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler(feature_range = (0,1))



scaler.fit(X)

X = scaler.transform(X)

X_test = scaler.transform(X_test)
from sklearn.linear_model import LogisticRegression

regressor=LogisticRegression()

regressor.fit(X,Y)

Y_test=regressor.predict(X_test)
result=pd.DataFrame({'PassengerId': Id,'Survived': Y_test})

result
result.to_csv('submission.csv',index=False)