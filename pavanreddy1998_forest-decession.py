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
df=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")
y=df["Cover_Type"]
x=df.iloc[:,:-1]
id=df.iloc[:,:1]
from sklearn.tree import DecisionTreeClassifier
reg=DecisionTreeClassifier()
reg.fit(x,y)
pred=reg.predict(test)
pred
id_pred=test.iloc[:,:1]
mysubmission=pd.DataFrame({'Id':test.Id,'Cover_Type':pred})
mysubmission=pd.DataFrame({'Id':test.Id,'Cover_Type':pred})
mysubmission.to_csv("submission.csv",index=False)
temp = pd.read_csv("submission.csv")
temp
