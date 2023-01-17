from IPython.display import Image
Image("../input/churnimage/churn.PNG")

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
data=pd.read_csv("../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Any results you write to the current directory are saved as output.



for i,v in enumerate(data.columns):
    print(i,v)
print("Total Number of Features {}".format(len(data.columns)))
data.dtypes
data.head(10)
def value_counts(column):
    plot=data[column].value_counts().plot.bar()
    print(data[column].value_counts())
    return plot
for i in data.columns[6:18]:
    value_counts(i)
    plt.show()

print("Percentage of Men  {}".format(3555/(3555+3488)*100))
print("Percentage of Women  {}".format(3488/(3555+3488)*100))


value_counts("SeniorCitizen")
value_counts("Dependents")
value_counts("Partner")
value_counts("PhoneService")
value_counts("MultipleLines")

data.dtypes
#value_counts("PaymentMethod")

plt.pie(x=[2365,1612,1544,1522],labels=["Electronic_Check","Mailed Check","Bank_Transfer","Credit Card"],autopct='%1.1f%%')




payment_gender=[]
for i,v in zip(data['gender'],data['PaymentMethod']):
    a=i,v
    payment_gender.append(a)
    
    

data['Payment_Gender']=payment_gender
data['Payment_Gender'].value_counts().plot.bar()
print(data.Payment_Gender.value_counts())
data.columns
data[['tenure','MonthlyCharges']]
plt.scatter(data['tenure'],data['MonthlyCharges'])

zero=data.loc[data['tenure']==0]

zero[['tenure','MonthlyCharges']]
zero
value_counts('Contract')
