# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid');

# Importing data
data = pd.read_csv('../input/kiva_loans.csv')
data.head()
data['sector'].value_counts().plot(kind="area",figsize=(12,12))
plt.xticks(np.arange(15), tuple(data['sector'].unique()), rotation=60)
plt.show()
data['repayment_interval'].value_counts().plot(kind="pie",figsize=(12,12))
f = sns.factorplot(x="sector",data=data,kind="count",hue="repayment_interval",size=12,palette="BuGn_r")
f.set_xticklabels(rotation=90)

print("Mean funded amount: ",data['funded_amount'].mean())
print("Mean lender count: ",data['lender_count'].mean())
print("Mean of term in months: ",data['term_in_months'].mean())
print("Correlation coefficent of funded amount and lender count: ",data['funded_amount'].corr(data['lender_count']))
sns.jointplot(x="funded_amount", y="lender_count", data=data, kind='reg')
print("Correlation coefficent of funded amount and term in months: ",data['funded_amount'].corr(data['term_in_months']))
sns.jointplot(x="funded_amount", y="term_in_months", data=data, kind='reg')
print("Correlation coefficent of term in months and lender count: ",data['term_in_months'].corr(data['lender_count']))
sns.jointplot(x="term_in_months", y="lender_count", data=data, kind='reg')
print(data['funded_amount'].corr(data['loan_amount']))
sns.jointplot(x="funded_amount", y="loan_amount", data=data, kind='reg')
data['month'] =  data['date'].astype(str).str[0:7]
data['month'].head()
pivot_data = pd.pivot_table(data,values='funded_amount',index='month',columns='sector')
pivot_data.head()
a4_dims = (15, 15)
fig, ax = plt.subplots(figsize=a4_dims)
sns.heatmap(pivot_data,ax=ax)
