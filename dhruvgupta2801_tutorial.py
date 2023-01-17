# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data=pd.read_csv('/kaggle/input/forest-cover-type-prediction/train.csv')

train_data.head()
test_data=pd.read_csv('/kaggle/input/forest-cover-type-prediction/test.csv')

test_data.head()
train_data.info()
test_data.info()
train_data['Slope'].plot(kind='hist')
test_data['Slope'].plot(kind='hist')
train_data['Elevation'].plot(kind='hist')
test_data['Elevation'].plot(kind='hist')
train_data['Cover_Type'].value_counts()
from sklearn.model_selection import train_test_split
X=train_data.drop(labels=['Id','Cover_Type'],axis=1)

y=train_data['Cover_Type']
X_train,X_val,y_train,y_val=train_test_split(X,y,random_state=40)
print(X_train.shape,y_train.shape)

print(X_val.shape,y_val.shape)
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=70)

rfc.fit(X_train,y_train)
rfc.score(X_val,y_val)
predict=rfc.predict(test_data.drop(labels=['Id'],axis=1))
Submission=pd.DataFrame(data=predict,columns=['Cover_Type'])

Submission.head()
Submission['Id']=test_data['Id']

Submission.set_index('Id',inplace=True)
Submission.head()
Submission.to_csv('Submission.csv')