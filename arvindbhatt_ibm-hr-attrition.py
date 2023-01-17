import pandas as pd

import numpy as np
import matplotlib.pyplot as plt


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
data=pd.read_csv('../input/WA_Fn-UseC_-HR-Employee-Attrition.csv')

# Any results you write to the current directory are saved as output.

import pandas as pd

import numpy as np
import matplotlib.pyplot as plt


data.groupby('EducationField').size()

data.hist(figsize=(15,15))
plt.show()
data.info()
data.describe()
print(data.head())
from sklearn.preprocessing import LabelEncoder
enc=LabelEncoder()
enc.fit(data.Attrition)
data.Attrition=enc.transform(data.Attrition)
enc.fit(data.BusinessTravel)
data.BusinessTravel=enc.transform(data.BusinessTravel)
enc.fit(data.Department)
data.Department=enc.transform(data.Department)
enc.fit(data.EducationField)
data.EducationField=enc.transform(data.EducationField)
enc.fit(data.Gender)
data.Gender=enc.transform(data.Gender)
enc.fit(data.JobRole)
data.JobRole=enc.transform(data.JobRole)
enc.fit(data.MaritalStatus)
data.MaritalStatus=enc.transform(data.MaritalStatus)
enc.fit(data.Over18)
data.Over18=enc.transform(data.Over18)
enc.fit(data.OverTime)
data.OverTime=enc.transform(data.OverTime)


data.head()
data.info()
Y=data.Attrition
data.drop(['Attrition'],axis=1,inplace=True)
from sklearn.preprocessing import StandardScaler
std=StandardScaler()
std.fit(data)
data=std.transform(data)

data=pd.DataFrame(data)
data.head()
data.corr()
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(  data, Y, test_size=0.33, random_state=42)

from sklearn.svm import SVC
model =SVC()
model.fit(X_train,y_train)
model.score(X_test,y_test)
from sklearn.linear_model import LogisticRegression
model1=LogisticRegression()
model1.fit(X_train,y_train)
model1.score(X_test,y_test)
