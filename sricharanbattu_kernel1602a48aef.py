# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
a=pd.read_csv('../input/train.csv')
b=pd.read_csv('../input/test.csv')
c=pd.read_csv('../input/gender_submission.csv')
print(a.head())
a.columns
a=a.replace('female',0)
a=a.replace('male',1)

features=['Pclass','Sex','Age']
sur=['Survived']
a['Age']=a['Age'].replace(np.nan,0)


z=np.array(a[['Age','Sex','Pclass']])
z
from sklearn.svm import SVC
svclassifier = SVC(kernel='rbf')  
svclassifier.fit(z,np.array(a['Survived']))  
b=b.replace('female',0)
b=b.replace('male',1)
b['Age']=b['Age'].replace(np.nan,0)
z1=np.array(b[features])


np.sum(svclassifier.predict(z)==np.array(a['Survived']))/len(z)
predictions=svclassifier.predict(z1)
print('variable predictions contain the test set predictions')