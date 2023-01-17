%matplotlib inline



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import seaborn as sns
data = pd.read_csv("../input/Salaries.csv")

data.info()
data = data.drop(["Id", "Notes", "Status", "Agency"], axis = 1)

data = data.dropna()

data.head()
columns = ["BasePay", "OvertimePay", "OtherPay", "Benefits", "TotalPay", "TotalPayBenefits"]

columns
data = data[data.BasePay != "Not Provided"]

data[columns] = data[columns].astype(np.float32)

data[columns].head()
# Delete rows which have BasePay <= 0

data.sort_values(by = columns, inplace = True)

data = data[data.BasePay > 0]



# Calculating (BasePay + OvertimePay + OtherPay) and (TotalPay + Benefits)

# SumedTotal = BasePay + OvertimePay + OtherPay

# SumedBenefits = TotalPay + Benefits

data["SumedTotal"] = data["BasePay"] + data["OvertimePay"] + data["OtherPay"]

data["SumedBenefits"] = data["TotalPay"] + data["Benefits"]



# Delete rows where SumedTotal != TotalPay and SumedBenefits != TotalPayBenefits

data = data[data.SumedTotal == data.TotalPay]

data = data[data.SumedBenefits == data.TotalPayBenefits]



data.head()
# Group by JobTitle to find most likely and unlikely job title

df_Job = data[columns + ["JobTitle"]].groupby(by = ["JobTitle"], as_index = False)

df_Job_NumberOfPeople = df_Job.count()

df_Job_NumberOfPeople["Count"] = df_Job_NumberOfPeople["BasePay"]

df_Job_NumberOfPeople.drop(columns, axis = 1, inplace = True)

df_Job_NumberOfPeople.sort_values(by = "Count", inplace = True)

df_Job_NumberOfPeople.head()
df_Job = df_Job.mean()



fig, (axis1,axis2) = plt.subplots(1,2,figsize=(20,5))



# 1] 10 job profile with highest BasePay 

# 2] 10 job profile with lowest BasePay

sns.barplot(x = "BasePay", y = "JobTitle", data = df_Job.sort_values(by = "BasePay", ascending = False).head(10), palette="Blues_d", ax = axis1)

sns.barplot(x = "BasePay", y = "JobTitle", data = df_Job.sort_values(by = "BasePay").head(10), palette="Blues_d", ax = axis2)



fig.tight_layout()
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(20,5))



# 1] 10 job profile with highest Benefits

# 2] 10 job profile with lowest Benefits

sns.barplot(x = "Benefits", y = "JobTitle", data = df_Job.sort_values(by = "Benefits", ascending = False).head(10), palette="Blues_d", ax = axis1)

sns.barplot(x = "Benefits", y = "JobTitle", data = df_Job.sort_values(by = "Benefits").head(10), palette="Blues_d", ax = axis2)



fig.tight_layout()
fig, ((axis1,axis2), (axis3,axis4)) = plt.subplots(2,2,figsize=(20,10))



# 1] 10 job profile with highest TotalPay and TotalPayBenefits

# 2] 10 job profile with lowest TotalPay and TotalPayBenefits

sns.barplot(x = "TotalPay", y = "JobTitle", data = df_Job.sort_values(by = "TotalPay", ascending = False).head(10), palette="Blues_d", ax = axis1)

sns.barplot(x = "TotalPay", y = "JobTitle", data = df_Job.sort_values(by = "TotalPay").head(10), palette="Blues_d", ax = axis2)



sns.barplot(x = "TotalPayBenefits", y = "JobTitle", data = df_Job.sort_values(by = "TotalPayBenefits", ascending = False).head(10), palette="Blues_d", ax = axis3)

sns.barplot(x = "TotalPayBenefits", y = "JobTitle", data = df_Job.sort_values(by = "TotalPayBenefits").head(10), palette="Blues_d", ax = axis4)



fig.tight_layout()
df_Year = data.groupby(by = "Year", as_index = False).mean()

df_Year
fig, ((axis1, axis2, axis3), (axis4, axis5, axis6)) = plt.subplots(2,3,figsize=(30,20))



# All six pay year wise

sns.barplot(x = "Year", y = "BasePay", data = df_Year, ax = axis1)

sns.barplot(x = "Year", y = "OvertimePay", data = df_Year, ax = axis2)

sns.barplot(x = "Year", y = "OtherPay", data = df_Year, ax = axis3)



sns.barplot(x = "Year", y = "Benefits", data = df_Year, ax = axis4)

sns.barplot(x = "Year", y = "TotalPay", data = df_Year, ax = axis5)

sns.barplot(x = "Year", y = "TotalPayBenefits", data = df_Year, ax = axis6)



fig.tight_layout()
df = data.groupby(by = ["Year", "JobTitle"], as_index = False).mean()

df[df.Year == 2012].min()