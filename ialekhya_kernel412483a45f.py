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
testset=pd.read_csv('/kaggle/input/summeranalytics2020/test.csv')
trainset=pd.read_csv('/kaggle/input/summeranalytics2020/train.csv')
ss=pd.read_csv('/kaggle/input/summeranalytics2020/Sample_submission.csv')

testset = testset.drop('EmployeeNumber', axis = 1)
trainset = trainset.drop('EmployeeNumber', axis = 1)
trainset.head(20)

from sklearn.preprocessing import LabelEncoder

for column in testset.columns:
        if testset[column].dtype == np.number:
            continue
        testset[column] = LabelEncoder().fit_transform(testset[column])
for column in trainset.columns:
        if trainset[column].dtype == np.number:
            continue
        trainset[column] = LabelEncoder().fit_transform(trainset[column])
xtest=testset
ytrain=trainset['Attrition']
xtrain=trainset.drop('Attrition',axis = 1)
from sklearn.linear_model import LogisticRegression  
Lr=LogisticRegression(solver='lbfgs',max_iter=5000,C=0.5,penalty='l2',random_state=1)
Lr.fit(xtrain,ytrain)
a=Lr.predict_proba(xtest)
b=a[:,1]
ss=ss.drop('Attrition',axis=1)
ss['Attrition']=b
ss.to_csv('submission.csv',index=False)
ss.head()