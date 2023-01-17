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

datatest=  pd.read_csv('/kaggle/input/summeranalytics2020/test.csv')
datatrain= pd.read_csv('/kaggle/input/summeranalytics2020/train.csv')
samplesub= pd.read_csv('/kaggle/input/summeranalytics2020/Sample_submission.csv')
datatest.head()
datatest= datatest.drop('EmployeeNumber', axis = 1)
datatest.head()
datatrain.head()
datatrain= datatrain.drop('EmployeeNumber', axis = 1)
datatrain.head()
samplesub= samplesub.drop(['Attrition'],axis=1)
from sklearn.preprocessing import LabelEncoder as label
from sklearn.linear_model import LogisticRegression as lr
for column in datatest.columns:
        if datatest[column].dtype == np.number:
            continue
        datatest[column] = label().fit_transform(datatest[column])
datatest.head(15)
for column in datatrain.columns:
        if datatrain[column].dtype == np.number:
            continue
        datatrain[column] = label().fit_transform(datatrain[column])
datatrain.head(15)
Y= datatrain ['Attrition']
X= datatrain.drop(['Attrition'], axis=1)
LogReg= lr(solver='lbfgs',max_iter=5000,C=0.5,penalty='l2',random_state=1)
LogReg.fit(X,Y)
z= LogReg.predict_proba(datatest)[:,1]
print(z)
samplesub['Attrition']=z
print(z)
samplesub.to_csv('sa2020submission.csv',index=False)
samplesub.head(25)