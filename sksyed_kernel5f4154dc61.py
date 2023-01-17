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
print(os.listdir("../input/credit-risk-modeling"))
pd_data=pd.read_csv("../input/credit-risk-modeling/train.csv")

pd_data.head(1)
def cleanTheData(pd_original_data):

    pd_data_Cleaning=pd_original_data

    #Cleaining Current Loan Amount

    convert_dict = {'Current Loan Amount': float} 

    pd_data_Cleaning=pd_data_Cleaning.astype(convert_dict)

    #Credit Score Cleaining

    pd_data_Cleaning['Credit Score'].fillna(pd_data_Cleaning['Credit Score'].mean(),inplace=True)

    #Annual Income

    pd_data_Cleaning['Annual Income'].fillna(pd_data_Cleaning['Annual Income'].median(),inplace=True)

    #Month Since Last Delinquent

    pd_data_Cleaning["Months since last delinquent"].fillna("0",inplace=True)

    convert_dict = {'Months since last delinquent': int} 

    pd_data_Cleaning=pd_data_Cleaning.astype(convert_dict)

    #Cleaining Years in Current Job

    mode=pd_data_Cleaning['Years in current job'].mode()

    pd_data_Cleaning['Years in current job'].replace('[^0-9]',"",inplace=True,regex=True)

    pd_data_Cleaning['Years in current job']=pd_data_Cleaning['Years in current job'].fillna(10)

    convert_dict = {'Years in current job': int} 

    pd_data_Cleaning= pd_data_Cleaning.astype(convert_dict)

    #Maximum Open Credit Cleaning

    pd_data_Cleaning["Maximum Open Credit"].replace('[a-zA-Z@_!#$%^&*()<>?/\|}{~:]',"0",regex=True,inplace=True)

    convert_dict = {'Maximum Open Credit': float} 

    pd_data_Cleaning=pd_data_Cleaning.astype(convert_dict)

    #BankRuptcies cleaining

    pd_data_Cleaning[pd_data_Cleaning.Bankruptcies.isna()==True]

    pd_data_Cleaning.Bankruptcies.fillna(0.0,inplace=True)

    convert_dict = {'Bankruptcies': int} 

    pd_data_Cleaning=pd_data_Cleaning.astype(convert_dict)

    #Tax Liens Cleaning

    pd_data_Cleaning['Tax Liens'].fillna(0.0,inplace=True)

    convert_dict = {'Tax Liens': int} 

    pd_data_Cleaning=pd_data_Cleaning.astype(convert_dict)

    #Monthly Debt Cleaning

    convert_dict = {'Monthly Debt': float} 

    pd_data_Cleaning["Monthly Debt"].replace('[^0-9.]',"",regex=True,inplace=True )

    pd_data_Cleaning["Monthly Debt"]=pd_data_Cleaning["Monthly Debt"].astype(convert_dict)

    

    return pd_data_Cleaning
pd_clean_Train_data=cleanTheData(pd_data)

pd_clean_Train_data.info()