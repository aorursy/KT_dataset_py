# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv("../input/years-of-experience-and-salary-dataset/Salary_Data.csv")
data.info()
data['Salary(in 1000s)']=data['Salary']/1000
data['yexpect']=data['Salary(in 1000s)']
data['xinit']=data['YearsExperience']
#calculating m
#Numerator part for OLS
avgx=data['YearsExperience'].mean(axis=0)
avgy=data['Salary(in 1000s)'].mean(axis=0)

errorsqinit=(data['Salary(in 1000s)']-avgy).pow(2).sum(axis=0)

data['YearsExperience']=data['YearsExperience']-avgx
data['Salary(in 1000s)']=data['Salary(in 1000s)']-avgy
data["mul"]=data['YearsExperience']*data['Salary(in 1000s)']
var1=data['mul'].sum(axis=0)

#denominator for OLS
var2=data['YearsExperience'].pow(2).sum(axis=0)


#m is:
m=var1/var2
#calculating b
b=avgy-(m*avgx)

#we have m(9.45) and b(25.79)

data['ypred']=m*data['xinit']+b
errorsqfin=(data['yexpect']-data['ypred']).pow(2).sum(axis=0)
#INTIAL ERROR SQUARE
print(errorsqinit)
#FINAL ERROR SQ VALUE AFTER LOG REG(MUCH LESSER)
print(errorsqfin)
data.head(5)
data.rename(columns={'yexpect':'Salary','xinit':'Years of Exp','ypred':'Predicted Salary'})

