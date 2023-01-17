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
df=pd.read_csv("../input/minor-project-2020/train.csv")
X = df.drop(['id','target'],axis=1)  
Y = df['target']     

from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=1)
X, Y = sm.fit_sample(X, Y)
from sklearn.preprocessing import StandardScaler;
sc = StandardScaler();
X = sc.fit_transform(X);
from sklearn.ensemble import RandomForestClassifier
cl=RandomForestClassifier()
cl.fit(X,Y)
dt=pd.read_csv("../input/minor-project-2020/test.csv")
T=dt.drop('id',axis=1)
Y=cl.predict_proba(T)[:,1]  
a = dt['id']
temp = []
for i in range(T.shape[0]):
    temp.append([a[i],Y[i]])
pd.DataFrame(temp).to_csv("./answer.csv", header=['id','target'],index=None)