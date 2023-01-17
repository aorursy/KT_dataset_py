# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
tn1=pd.read_csv('../input/train.csv')
tn1=tn1.drop('maxPlace',axis=1)
trainx=tn1.drop('winPlacePerc',axis=1)
trainy=tn1['winPlacePerc']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(trainx,trainy,random_state=101,test_size=0.7)
from sklearn.linear_model import LogisticRegression
lg=LogisticRegression()
from sklearn.metrics import confusion_matrix
y_trainR=pd.Series([1 if i<=0.6 else 0 for i in y_train])
y_testR=pd.Series([1 if i<=0.6 else 0 for i in y_test])
model=lg.fit(x_train,y_trainR)
res=model.predict(x_test)
print(confusion_matrix(y_testR,res))
testD=pd.read_csv('../input/test.csv')
testD=testD.drop('maxPlace',axis=1)
res2=model.predict(testD)
result=pd.DataFrame(testD['Id'])
final=result.assign(Predict=res2)