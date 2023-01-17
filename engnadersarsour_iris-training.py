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
path='../input/iris/Iris.csv'

data=pd.read_csv(path,index_col='Id')
data.head()
data.columns
data.isna().sum()
#data.drop(columns=['Name_coulm','Ticket','Cabin'],inplace=True)
mean_val=data.SepalLengthCm.mean()

mean_val
data.SepalLengthCm.fillna(mean_val,inplace=True)
data.Species.unique() 
data.Species.replace('Iris-setosa',0,inplace=True)

data.Species.replace('Iris-versicolor',1,inplace=True)

data.Species.replace('Iris-virginica',2,inplace=True)



data.head()
from sklearn.model_selection import train_test_split

X=data.drop(columns='Species',)

y=data.Species

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
from sklearn.tree import DecisionTreeClassifier
data_model=DecisionTreeClassifier()

data_model.fit(X_train,y_train)
prediction=data_model.predict(X_test)
from sklearn.metrics import mean_absolute_error,accuracy_score,classification_report
error=mean_absolute_error(y_test,prediction)

accuracy_score(y_true=y_test,y_pred=prediction)
error
print(classification_report(y_true=y_test,y_pred=prediction))