import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
##The data set of an international bank was obtained from Kaggle  

##https://www.kaggle.com/nasirislamsujan/bank-customer-churn-prediction

df = pd.read_csv("../input/deep-learning-az-ann/Churn_Modelling.csv")
#Information about the data set

df.info()
##First five rows and all columns

df.head()
##Dropping RowNumber, CustomerId, and Surname columns. 

df.drop(["RowNumber","CustomerId","Surname"], axis=1, inplace=True)

df.describe()
#Chechking NA values

df.isna().sum()
##Changing the name of Exited column as Churn 

df.loc[(df['Exited'] == 0), 'Churn'] = 'Non-Churn'

df.loc[(df['Exited'] == 1), 'Churn'] = 'Churn'

df.head()
##Grouping ages to investigate age group effect on the churn decision 

## Student: under Age 25, Employee: Age 25- 64, Retired: Age 65 and older

df.loc[(df['Age'] < 25), 'AgeGroup'] = 'Under age 25'

df.loc[(df['Age'] >= 25) & (df['Age']<= 64), 'AgeGroup'] = 'Age 25-64'

df.loc[(df['Age'] >= 65), 'AgeGroup'] = 'Age 65 and older'

df.head()
freq=pd.crosstab(df.Geography, df.Churn)

freq['Churn'].plot(kind='pie', autopct='%1.1f%%')
mask=(df.Churn=='Churn')

freq=pd.crosstab(df[mask].Geography, df[mask].NumOfProducts)

sns.heatmap(freq, cmap="YlGnBu", annot=True, cbar=True, fmt='d')

plt.title('Churned Customers',fontsize = 12, weight='bold')

plt.ylabel("")
freq=pd.crosstab(df.Gender, df.Churn)

freq['Churn'].plot(kind='pie', autopct='%1.1f%%')
mask=(df.Churn=='Churn')

freq=pd.crosstab(df[mask].Geography, df[mask].Gender)

freq.plot(kind='bar')
sns.heatmap(pd.crosstab(df.Tenure, df.Churn), cmap="YlGnBu", annot=True, cbar=True, fmt='d')
sns.heatmap(pd.crosstab(df.NumOfProducts, df.Churn), cmap="YlGnBu", annot=True, cbar=True, fmt='g')
freq=pd.crosstab(df.HasCrCard, df.Churn)

freq.plot(kind='bar')
freq=pd.crosstab(df.IsActiveMember, df.Churn)

freq.plot(kind='bar')
mask=(df.Churn=='Churn')

sns.violinplot(x="Tenure", y="CreditScore", data=df[mask])
plt.figure(figsize=[8,10])

plt.subplot(3,1,1)

mask1=(df.Geography=='Germany')

sns.swarmplot(x="AgeGroup", y="CreditScore", hue="Churn", data=df[mask1], size=6, palette="Set2", dodge=True, 

              order=["Under age 25", "Age 25-64", "Age 65 and older"])

plt.xlabel("")

plt.ylabel('Credit Scores',fontsize = 11, style='italic', weight='bold')

plt.title('GERMANY',fontsize = 12, weight='bold')

plt.legend(['Non-Churn', 'Churn'], ncol=2, loc='lower right')

plt.xticks(size = 10)

plt.yticks(size = 10)



plt.subplot(3,1,2)

mask2=(df.Geography=='France')

sns.swarmplot(x="AgeGroup", y="CreditScore", hue="Churn", data=df[mask2], size=6, palette="Set2", dodge=True,

             order=["Under age 25", "Age 25-64", "Age 65 and older"])

plt.xlabel("")

plt.ylabel('Credit Scores',fontsize = 11, style='italic', weight='bold')

plt.title('FRANCE',fontsize = 12, weight='bold')

plt.legend(['Non-Churn', 'Churn'], ncol=2, loc='lower right')

plt.xticks(size = 10)

plt.yticks(size = 10)



plt.subplot(3,1,3)

mask3=(df.Geography=='Spain')

sns.swarmplot(x="AgeGroup", y="CreditScore", hue="Churn", data=df[mask3], size=6, palette="Set2", dodge=True,

              order=["Under age 25", "Age 25-64", "Age 65 and older"])

plt.ylabel('Credit Scores',fontsize = 11, style='italic', weight='bold')

plt.title('SPAIN',fontsize = 12, weight='bold')

plt.legend(['Non-Churn', 'Churn'], ncol=2, loc='lower right')

plt.xticks(size = 10)

plt.yticks(size = 10);

plt.plot()
plt.figure(figsize=[8,10])

plt.subplot(3,1,1)

mask1=(df.Geography=='Germany')

sns.swarmplot(x="AgeGroup", y="Balance", hue="Churn", data=df[mask1], size=6, dodge=True,

             order=["Under age 25", "Age 25-64", "Age 65 and older"])

plt.xlabel("")

plt.ylabel('Balance',fontsize = 11, style='italic', weight='bold')

plt.title('GERMANY',fontsize = 12, weight='bold')

plt.legend(['Non-Churn', 'Churn'], ncol=2, loc='upper right')

plt.xticks(size = 10)

plt.yticks(size = 10)



plt.subplot(3,1,2)

mask2=(df.Geography=='France')

sns.swarmplot(x="AgeGroup", y="Balance", hue="Churn", data=df[mask2], size=6, dodge=True,

             order=["Under age 25", "Age 25-64", "Age 65 and older"])

plt.xlabel("")

plt.ylabel('Balance',fontsize = 11, style='italic', weight='bold')

plt.title('FRANCE',fontsize = 12, weight='bold')

plt.legend(['Non-Churn', 'Churn'], ncol=2, loc='upper right')

plt.xticks(size = 10)

plt.yticks(size = 10)



plt.subplot(3,1,3)

mask3=(df.Geography=='Spain')

sns.swarmplot(x="AgeGroup", y="Balance", hue="Churn", data=df[mask3], size=6, dodge=True,

             order=["Under age 25", "Age 25-64", "Age 65 and older"])

plt.xlabel("")

plt.ylabel('Balance',fontsize = 11, style='italic', weight='bold')

plt.title('SPAIN',fontsize = 12, weight='bold')

plt.legend(['Non-Churn', 'Churn'], ncol=2, loc='upper right')

plt.xticks(size = 10)

plt.yticks(size = 10)

plt.plot();
plt.figure(figsize=[10,4])

mask=(df.HasCrCard==0) & (df.IsActiveMember==0)

sns.violinplot(x="NumOfProducts", y="Age", hue="Churn", data=df[mask], palette="Blues")

plt.xlabel("NumOfProducts")

plt.ylabel("Age")

plt.title('HasCreditCard=0 | IsActiveMember=0',fontsize = 12)

plt.xticks(size = 10)

plt.yticks(size = 10);
plt.figure(figsize=[10,4])

mask=(df.HasCrCard==1) & (df.IsActiveMember==1)

sns.violinplot(x="NumOfProducts", y="Age", hue="Churn", data=df[mask], palette="Greens")

plt.xlabel("NumOfProducts")

plt.ylabel("Age")

plt.title('HasCreditCard=1 | IsActiveMember=1',fontsize = 12)

plt.xticks(size = 10)

plt.yticks(size = 10);