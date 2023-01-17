# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd# data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import matplotlib.pyplot as plt
from fancyimpute import KNN
import seaborn as sns
# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/train_AV3.csv")
test=pd.read_csv("../input/test_AV3.csv")
f=data.append(test,sort=False) 
f.dtypes






#data.duplicated().sum()
f.duplicated().sum()
# There is no duplicate data present


data.isnull().sum()
#to find the variables having missing values in train db
f.isnull().sum()
# it gives insight that the missing values of the dependent variable i.e. Loan_Status is missing only for train db
f.shape # of which 614 belongs to train db
f.describe()
def f1(x):
    if(x<=2875):
        return "L"
    elif(x<=3800):
        return "LM"
    elif(x<=5516):
        return "UM"
    else:
        return "U"
    

f["ApplicantStatus"]=f["ApplicantIncome"].map(f1)
f.drop(["Loan_ID"],axis=1)

  #a new column is created to categorize the apllicant's income in the four quartiles as L,LM,UM,U(Lower,LowerMiddle,UpperMiddle,Upper)  
f["LoanAmount"].fillna(f["LoanAmount"].mean(),inplace=True)
f["Loan_Amount_Term"].fillna(360.0,inplace=True) #imputing with mode value
#s=np.column_stack((f["ApplicantIncome"],f["Credit_History"]))
#t=KNN(k=4).complete(s)


datac=data[data["Credit_History"]==0]
datac[datac["Loan_Status"]=='Y']

#there are only 7 exceptional cases where loan has been approved despite credit history not meeting the requirements.
# lets assume that the ids for which credit history is missing meet the requirements
f["Credit_History"].fillna(1.0,inplace=True)
 
f[f["Credit_History"]==1.0].groupby(['Gender','Loan_Status']).size() #annalyzing the loan applicants who meets the requirements

# in general the male applicants have a higher chance than the female applicants.
fm=f[f["Gender"]=='Male']
ff=f[f["Gender"]=='Female']

ff[ff["Credit_History"]==1.0].groupby(['ApplicantStatus','Loan_Status']).size()
# l-74 lm-86 um-68 u-83
ff[ff["Credit_History"]==1.0].groupby(['ApplicantStatus','Loan_Status']).size()




