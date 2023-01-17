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
import numpy as np
import pandas as pd
#data = pd.read_csv('train.csv')
data = pd.read_csv("/kaggle/input/womenintheloop-data-science-hackathon/train.csv")
data1 = pd.read_csv("/kaggle/input/womenintheloop-data-science-hackathon/test_QkPvNLx.csv")
del data['User_Traffic']
Y = data.iloc[:,-1]
data = data.iloc[:,0:data.shape[1]-1];

Course_Domain = {'Development': 1,'Software Marketing': 2,'Finance & Accounting' : 3,'Business' : 4 }
data.Course_Domain = [Course_Domain[item] for item in data.Course_Domain] 
Course_Domain = {'Development': 1,'Software Marketing': 2,'Finance & Accounting' : 3,'Business' : 4 }
data1.Course_Domain = [Course_Domain[item] for item in data1.Course_Domain] 
Course_Type  = {'Course': 5,'Program': 6,'Degree' : 7 }
data.Course_Type  = [Course_Type [item] for item in data.Course_Type ] 
Course_Type  = {'Course': 5,'Program': 6,'Degree' : 7 }
data1.Course_Type  = [Course_Type [item] for item in data1.Course_Type ] 
data["Competition_Metric"] = data["Competition_Metric"].fillna(data["Competition_Metric"].mean())
data1["Competition_Metric"] = data1["Competition_Metric"].fillna(data1["Competition_Metric"].mean())
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 1000, random_state = 1,max_depth = 6)
rf.fit(data, Y);
rf.score(data,Y)
from sklearn.linear_model import LinearRegression
from sklearn import metrics
rf = LinearRegression()  
rf.fit(data, Y) #training the algorithm
#To retrieve the intercept:
print(rf.intercept_)
#For retrieving the slope:
print(rf.coef_)
y_pred = rf.predict(data1)
a = data1['ID']
df = pd.DataFrame({"ID" : a, "sale" : y_pred})
df.to_csv("sample.csv", index=False)

rf.score(data,Y)