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
%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
df=pd.read_csv("../input/TitanicDataset/titanic_data.csv")
miss=df[df['Age'].isnull()]
miss
new_df=df[{'Survived','Pclass','Sex','Age','Fare'}]
dumy1=pd.get_dummies(new_df.Pclass)
dumy1.head(3)
dumy2=pd.get_dummies(new_df.Sex)
dumy2.head()
new_df=pd.concat([new_df,dumy1,dumy2],axis=1)
new_df.drop(['Pclass','Sex'],axis=1,inplace=True)
new_df['Age']=new_df['Age'].replace(0,np.NaN)
mean=int(new_df['Age'].mean(skipna=True))
new_df['Age']=new_df['Age'].replace(np.NaN,mean)
new_df.columns[new_df.isna().any()]
new_df
X=new_df.iloc[:,1:].values
y=new_df.iloc[:,0].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
model=GaussianNB()
model.fit(X_train,y_train)
model.score(X_test,y_test)