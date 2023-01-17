import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import os

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/WA_Fn-UseC_-HR-Employee-Attrition.csv")

data.info()
data.describe()
f,ax = plt.subplots(figsize=(30,30))

sns.heatmap(data.corr(),annot=True,linewidth=.3,fmt='.2f')

plt.show()
data.head(10)
data.plot(kind='scatter', x='MonthlyIncome', y='YearsAtCompany', alpha = 0.5,color = 'red',figsize=(15,5))

plt.xlabel('Income')              

plt.ylabel('Years at Company')

plt.title('Income-Years at Company Scatter Plot')           
data.Age.plot(kind = 'hist',bins = 40,figsize = (20,15))

plt.show()
x = data['JobRole']=='Manufacturing Director'

data[x]
y = data['Age']<30

data[x&y]            #shows information of manufacturing directors who are below 30
managers = data['JobRole']=='Manager'

data[managers].Education.plot(kind='hist',bins=4,figsize=(10,10))
c1=0

c2=0

c3=0

c4=0

c5=0

c6=0

for index,value in data[['EducationField']].iterrows():

    if (value.item()=='Human Resources'):

        c1 = c1 + 1

    elif (value.item()=='Life Sciences'):

        c2 = c2 + 1

    elif (value.item()=='Marketing'):

        c3 = c3 + 1

    elif (value.item()=='Medical'):

        c4 = c4 + 1

    elif (value.item()=='Technical Degree'):

        c5 = c5 + 1

    elif (value.item()=='Other'):

        c6 = c6 + 1

print("Education field of ",c1,"employees is Human Resources.")

print("Education field of ",c2,"employees is Life Sciences.")

print("Education field of ",c3,"employees is Marketing.")

print("Education field of ",c4,"employees is Medical.")

print("Education field of ",c5,"employees is Technical Degree.")

print("Education field of ",c6,"employees is none of above.")

 
i = 0

n1=0

n2=0

n3=0

a = 0  #will be calculated as number of employees doing overtime



while (i < len(data.OverTime)):

    if (data.OverTime[i]=='Yes'):

        if (data.MonthlyIncome[i] <= 2500):

            n1 = n1 + 1

        elif (2500 < data.MonthlyIncome[i] <= 6500):

            n2 = n2 + 1

        elif (data.MonthlyIncome[i] > 6500):

            n3 = n3 + 1

        a = a + 1

    i = i + 1

    

r1 = round((n1/a),2)

r2 = round((n2/a),2)

r3 = round((n3/a),2)

    

print(r1," of employees doing overtime have monthly income lower than 2500.")

print(r2," of employees doing overtime have monthly income between 2500 and 6500.")

print(r3," of employees doing overtime have monthly income greater than 6500.")

    
lacking = []    

i = 0

while(i < len(data.EmployeeNumber)-1):    

    if (data.EmployeeNumber[i+1] != data.EmployeeNumber[i] + 1):

        diff = data.EmployeeNumber[i+1] - data.EmployeeNumber[i]   

        k = 1

        c = data.EmployeeNumber[i]+1

        while (k < diff):           

            lacking.append(c)

            c = c + 1

            k = k + 1

    i= i + 1

    

lacking