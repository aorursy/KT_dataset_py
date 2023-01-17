# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random as rd
from sklearn.linear_model import LogisticRegression as lg
from sklearn.preprocessing import MinMaxScaler
rd.seed(18)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv("../input/minor-project-2020/train.csv",header=0)
ID = data['id']
y = data['target']
data.drop(['id','target'],axis=1, inplace = True)


logReg=lg(max_iter=1500,class_weight="balanced", solver = 'saga')
logReg.fit(data,y)



testX=pd.read_csv("../input/minor-project-2020/test.csv")
id_=testX['id']
testX.drop(['id'],axis=1, inplace = True)
y_temp=logReg.predict_proba(testX)

id_=id_[:,np.newaxis]
y_pred_test = (y_temp)[:,1:2]
res=np.concatenate([id_,y_pred_test],axis=1)
dataFrame=pd.DataFrame(data=res,columns=["id","target"])
dataFrame.to_csv("2018A7PS0090G.csv")


