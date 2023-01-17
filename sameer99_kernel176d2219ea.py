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
import pandas as pd
from sklearn import linear_model
data=pd.read_csv("../input/train.csv")
xtrain=data.iloc[:,1:].values
y=data.iloc[:,0].values
clf=linear_model.LogisticRegression()

clf.fit(xtrain,y)

test=pd.read_csv("../input/test.csv")
xtest=test.iloc[:,:].values
result=clf.predict(xtest)
result

id=list()
label=list()

for x,y in  enumerate(result):
    id.append(x+1)
    label.append(y)
    
final=pd.DataFrame({'ImageId':id,'Label':label})
final
final.to_csv("Submit.csv")
