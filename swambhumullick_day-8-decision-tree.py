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

import numpy as np

data = pd.read_csv("/kaggle/input/apndcts/apndcts.csv")

data.head()
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix
predictors=data.iloc[:,0:7]

target=data.iloc[:,7]
predictors
target
P_train,P_test,T_train,T_test = train_test_split(predictors,target,test_size=0.3,random_state= 125)

d_entropy = DecisionTreeClassifier(criterion="entropy",random_state= 100,max_depth=3,min_samples_leaf=5)

model=d_entropy.fit(P_train,T_train)

predictions=d_entropy.predict(P_test)
print('accuracy score :',accuracy_score(T_test,predictions,normalize=True))

print('confusion matrix : \n ',confusion_matrix(T_test,predictions))