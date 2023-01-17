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
from sklearn.svm import SVC
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
train_data=pd.read_csv("train.csv")

test_data=pd.read_csv("test.csv")
train_data = train_data.fillna(method='ffill')
test_data = test_data.fillna(method='ffill')
# print(train_data.iloc[:,0:21])
train=np.array(train_data.iloc[:,2:21]).astype(np.float64)
train_labels=np.array(train_data.iloc[:,21:]).astype(np.float64)

test=np.array(test_data.iloc[:,2:21])
test_labels=np.array(test_data.iloc[:,21:])

clf = SVC()
clf.fit(train,train_labels)

prediction=clf.predict(test)
ids=np.array(test_data.iloc[:,0])

prediction=prediction.astype(np.int)

print(np.column_stack((ids,prediction)))

cols = { 'PlayerID': ids , 'TARGET_5Yrs': prediction }

result=pd.DataFrame(cols)
print(result)
result.to_csv("submission.csv", index=False)