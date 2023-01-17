import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import os

print(os.listdir("../input"))
data = pd.read_csv("../input/WA_Fn-UseC_-HR-Employee-Attrition.csv")

data.info()
data.columns
a = data.YearsInCurrentRole[data.YearsAtCompany>2]   #a and b includes employees who work at the company for more than 2 years

b = data.YearsAtCompany[data.YearsAtCompany>2]

z = list(map(lambda x, y: x/y, a,b))



i=0

c1=0

c2=0

while(i<len(z)):

    if (z[i]<=0.5):

        c1 = c1 + 1

    elif (z[i]>0.5):

        c2 = c2 + 1

    i = i + 1

    

c1,c2
data["Experience"]=["Entry Level" if data.TotalWorkingYears[i]<=3 else "Specialist" if 3<data.TotalWorkingYears[i]<10 else "Senior" for i in data.TotalWorkingYears]

data.loc[:10,["Experience","Age"]]
print(data["Department"].value_counts(dropna=False))       #let's look at departments these employees work at and education fields of them

print(data["EducationField"].value_counts(dropna=False))
data.boxplot(column="MonthlyIncome", by="Gender", figsize=(10,15))
travel_department = pd.melt(frame=data, id_vars="EmployeeNumber", value_vars=["Department","BusinessTravel"])

travel_department[travel_department.EmployeeNumber < 20]   #let's look at some of them
travel = data.BusinessTravel.head(15)

department = data.Department.head(15)

travel_dept = pd.concat([travel,department],axis=1)

travel_dept     #the same data above, more easy to examine by this way