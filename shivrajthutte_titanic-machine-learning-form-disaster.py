# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

train = pd.read_csv("/kaggle/input/titanic/train.csv")

train.head()
median_value=train['Age'].median()
train['Age']=train['Age'].fillna(median_value)
train=train.drop(["PassengerId","Name","Ticket","Cabin"],axis=1)
train1=pd.get_dummies(train,drop_first=True)
X=train1.iloc[:,1:9]

y=train1.iloc[:,:1]
test=pd.read_csv("/kaggle/input/titanic/test.csv")

test.head()
median_value=test['Age'].median()

test['Age']=test['Age'].fillna(median_value)
test=test.drop(["Name","Ticket","Cabin"],axis=1)
test1=pd.get_dummies(test,drop_first=True)
TestID = test1["PassengerId"]
PassengerId = pd.DataFrame(TestID)

test1=test1.iloc[:,1:9]
from xgboost import XGBClassifier

clf=XGBClassifier(random_state=2020,n_jobs=-1)

clf.fit(X,y)
import numpy as np

y_pred_prob=clf.predict_proba(test1)[:,1]

sur=np.where(y_pred_prob>0.5,1,0)
PassengerId = PassengerId['PassengerId']

Survived = sur
submit = pd.DataFrame({'PassengerId':PassengerId, 'Survived':Survived})