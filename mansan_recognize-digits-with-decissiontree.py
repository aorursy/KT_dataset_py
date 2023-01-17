# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/train.csv')
data.head(4)
data.isnull().sum()
X=data.iloc[:,data.columns!='label']
Y=data.iloc[:,data.columns=='label']

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=6)
X_train.shape
model=DecisionTreeClassifier(criterion = "entropy", random_state = 6)
model.fit(X_train,Y_train)
Predict=model.predict(X_test)
Output=pd.DataFrame(Predict,columns=['Predicted'])
Output['Actual']=Y_test.reset_index()['label']
Output['Predict_Status']='Correct'
Output['Predict_Status']=Output['Predict_Status'].where(Output['Predicted'] == Output['Actual'],'Wrong')
Output.head()
Output.groupby('Predict_Status').Predict_Status.count()
sns.countplot(x='Predict_Status',data=Output, palette='hls')
data_test=pd.read_csv('../input/test.csv')
data_test.head()
Predict_Test=model.predict(data_test)
Final_Test=pd.DataFrame(Predict_Test,columns=['Label'])
Final_Test.index.name='ImageID'
Final_Test.index += 1
Final_Test.to_csv('Submission.csv')
