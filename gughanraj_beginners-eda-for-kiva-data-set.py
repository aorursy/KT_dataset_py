# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import time
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
loans = pd.read_csv("../input/kiva_loans.csv")
loans.columns
loans.head(2)
#Missing Value Identification 
pd.concat([loans.isnull().sum().sort_values(ascending=False)])
plt.subplots(figsize=(15,8))
plt.xticks(rotation='vertical')
sns.countplot(x="sector",data=loans,orient="h")
# Top Activity for which loans used for 
agg1 = pd.DataFrame(loans.activity.value_counts().head(15))
agg1= agg1.reset_index()
agg1.columns = ['Activity','Count']
plt.figure(figsize=(15,6))
plt.xticks(rotation="vertical")
sns.barplot(x="Activity",y="Count",data=agg1)
#Top Countries with most loans
agg1 = pd.DataFrame(loans.country.value_counts().head(15))
agg1= agg1.reset_index()
agg1.columns = ['Country','Count']
plt.figure(figsize=(15,6))
sns.barplot(x="Country",y="Count",data=agg1)
loans.repayment_interval.value_counts().plot(kind="pie",figsize=(6,6),autopct='%1.1f%%')
print("More than 50% of loans opted for Monthly repayment mode")
def gender(x):
    lis = x.split(",")
    lis = list(set(lis))
    lis = [x.strip() for x in lis]
    if len(lis)==2:
        return "Both"
    elif lis[0]=="female":
        return "female"
    else:
        return "male"

top_g = loans.borrower_genders.value_counts().reset_index().head(1)['index'][0]
loans.borrower_genders[loans.borrower_genders.isnull()]= top_g
loans['gender1'] = loans.borrower_genders.apply(gender)
loans.gender1.value_counts().plot(kind="pie",autopct="%1.1f%%",figsize=(6,6))
def gender_cnt(x):
    lis = x.split(",")
    return len(lis)

loans['borrower_count'] = loans.borrower_genders.apply(gender_cnt)
plt.figure(figsize=(15,6))
sns.distplot(loans.borrower_count,bins=20)
plt.figure(figsize=(15,6))
plt.xlim([0,100])
plt.ylabel("Count")
sns.distplot(loans.term_in_months,kde=False)
sns.boxplot(data=loans[['funded_amount','loan_amount']],orient="h")
loans['funded_date'] = pd.to_datetime(loans.date,errors='coerce')
loans['Year']  = loans.funded_date.apply(lambda x : x.year)
loans['Month'] = loans.funded_date.apply(lambda x : x.month)
#Heat map to visualize the loan application count 
time_agg = loans.groupby(['Year','Month'],as_index=False).size().reset_index()
time_agg.columns = ['Year','Month','loan_count']
time_agg = time_agg.pivot(index='Month',columns='Year',values='loan_count')
time_agg = time_agg.fillna(0)
plt.figure(figsize=(12,6))
sns.heatmap(time_agg,cmap="YlGnBu")
