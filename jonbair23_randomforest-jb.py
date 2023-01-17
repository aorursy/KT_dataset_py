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
import math 
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
files = os.listdir("../input")
x_file = files[3]
y_file = files[2]
Xs = pd.read_csv("../input/" + x_file)
Ys = pd.read_csv("../input/" + y_file)
Xs = Xs.dropna(axis = 1)
X_columns = Xs.columns 
Xs = Xs.drop(columns=[X_columns[0],X_columns[1],X_columns[2],X_columns[3], X_columns[4], X_columns[5]])
Xs = Xs.drop(columns=['EmployeeID', 'RatingTableID'])
Xs = Xs.drop(columns=['CNVersion'])
Xs; 
Ys_Overall = Ys['OverallScore']
Ys_Overall; 
Xs = Xs.drop(columns=['Tax_Year'])
vals = ['Creation_Date','date_calculated','publish_date','Rpt_Comp_Date','Rpt_Ap_Date','Rpt_Ap_Emp','Govt_Grants','Total_Contributions','ProgSvcRev','MemDues','PrimaryRev','Other_Revenue','Total_Revenue','Excess_Deficit','Total_Expenses','Program_Expenses','Administration_Expenses','Fundraising_Expenses','Pymt_Affiliates','Total_Assets','Total_Liabilities','Total_Net_Assets','Total_Func_Exp']
Ys_Overall = Ys_Overall[np.isfinite(Ys_Overall)]

Xs = Xs.loc[Ys_Overall[np.isfinite(Ys_Overall)].index]
trainx, xtestvals,trainy,ytestvals = train_test_split(Xs, Ys_Overall, test_size = 0.20, random_state=42)

means = trainx[vals].mean()
stds = trainx[vals].std()
trainx[vals] = (trainx[vals]-means)/stds
def rmse(p,targets): 
    return np.sqrt((p - targets)**2).mean() 
from sklearn.ensemble import RandomForestRegressor 


xtestvals[vals]=(xtestvals[vals]-means)/stds

num = 20
min_error = 100000000
while (num < 100): 
    forest = RandomForestRegressor(n_estimators=num, random_state=0).fit(trainx,trainy)
    predictions = forest.predict(xtestvals)
    error = rmse(predictions, ytestvals) 
    if (error < min_error): 
        print(str(error) + " " + str(num))
        min_error = error 
    num = num + 10 

    
    
forr = RandomForestRegressor(n_estimators=80, random_state=0).fit(trainx,trainy)
test_data = pd.read_csv("../input/testFeatures.csv")
test_data = test_data[trainx.columns]
test_vals = test_data.copy()
test_vals[vals] = (test_data[vals]-means)/stds
test_vals.head()
pred = forr.predict(test_vals)
output=pd.DataFrame({'Id':np.arange(1,2127),'OverallScore':pred})

output.to_csv('out_randomforest.csv', index = False)
