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
from sklearn import preprocessing
training= pd.read_csv('/kaggle/input/summeranalytics2020/train.csv')
test=pd.read_csv('/kaggle/input/summeranalytics2020/test.csv')
df=training.drop(columns=['Attrition'],axis=1)
X=pd.get_dummies(df)
y=training['Attrition']
dt=pd.get_dummies(test)

from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
#X, y = make_classification(random_state=0)
clf = GradientBoostingClassifier()
clf.fit(X, y)

clf.score(X,y)
res=clf.predict_proba(dt)
res1=res[:,1]
res1
test['Attrition']=res1
test[['Id','Attrition']].to_csv('results_with_boost.csv',index=False)
