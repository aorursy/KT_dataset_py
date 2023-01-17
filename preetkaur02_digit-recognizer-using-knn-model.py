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
train_imp = pd.read_csv('../input/train.csv')
test_imp = pd.read_csv('../input/test.csv')
from sklearn.model_selection import train_test_split
train,test=train_test_split(train_imp,test_size=0.3,random_state=100)
train_x = train.drop('label', axis=1)
train_y = train['label']

test_x = test.drop('label',axis=1)
test_y =test['label']

from sklearn.neighbors import KNeighborsClassifier
model_imp=KNeighborsClassifier(n_neighbors=5)
model_imp.fit(train_x,train_y)
pred_test=model_imp.predict(test_x)
from sklearn.metrics import accuracy_score
accuracy_score(test_y,pred_test)
pred_test=model_imp.predict(test_imp)
df=pd.DataFrame(pred_test,columns=['Label'])
df['ImageId']=test_imp.index+1
df[['ImageId','Label']].to_csv('submission.csv',index=False)
pd.read_csv('submission.csv').head()
