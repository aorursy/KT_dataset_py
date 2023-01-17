# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns 
import matplotlib.pyplot as plt 
%matplotlib inline
import missingno as msno
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

df_loans = pd.read_csv("../input/kiva_loans.csv")
df_region_locations = pd.read_csv("../input/kiva_mpi_region_locations.csv")
df_theme_ids = pd.read_csv("../input/loan_theme_ids.csv")
df_themes_by_region = pd.read_csv("../input/loan_themes_by_region.csv")
# Any results you write to the current directory are saved as output.
msno.matrix(df_loans)
df_loans['posted_time'] = pd.to_datetime(df_loans['posted_time'])
df_loans['disbursed_time'] = pd.to_datetime(df_loans['disbursed_time'])
df_loans['funded_time'] = pd.to_datetime(df_loans['funded_time'])
df_loans['date'] = pd.to_datetime(df_loans['date'])
df_loans.head(2)
df_loans['loan_amount'].describe()
plt.figure(figsize=(12,10))
sns.distplot(df_loans['loan_amount'],bins=1000)
plt.title("Loan Amount Distribtion")
plt.xlabel("Loan Amount(Dollars)")
plt.xlim(0,10000)
print("2 Standard deviations from the mean :%s" %(round(df_loans['loan_amount'].std())*2))
plt.figure(figsize=(12,8))
sns.distplot(df_loans.query("loan_amount<2398")['loan_amount'])
sector_wise_amounts = df_loans.groupby("sector").mean()[["funded_amount"]].sort_values(ascending=False,by="funded_amount")
sector_wise_amounts.reset_index()
sector_wise_amounts = sector_wise_amounts.T.squeeze()

plt.figure(figsize=(12,8))
sns.barplot(y=sector_wise_amounts.index, x=sector_wise_amounts.values, alpha=0.6)
plt.ylabel("Sectors")
plt.xlabel("Funded Amounts")
plt.title("Distribution of Funded Amounts by Sectors")
print("Percent of the dataset below 800$ Funded amount :{:0.2f} % " .format(100*len(df_loans.query("funded_amount<=800"))/len(df_loans)))
df_business_classA = df_loans.query("funded_amount>850")
df_business_classB = df_loans.query("funded_amount<=850")
sector_wise_amounts = df_business_classA.groupby("sector").mean()[["funded_amount"]].sort_values(ascending=False,by="funded_amount")
sector_wise_amounts.reset_index()
sector_wise_amounts = sector_wise_amounts.T.squeeze()

plt.figure(figsize=(12,8))
sns.barplot(y=sector_wise_amounts.index, x=sector_wise_amounts.values, alpha=0.6)
plt.ylabel("Sectors")
plt.xlabel("Funded Amounts")
plt.title("Distribution of Funded Amounts by Sectors in Business Type A")
sector_wise_amounts = df_business_classB.groupby("sector").mean()[["funded_amount"]].sort_values(ascending=False,by="funded_amount")
sector_wise_amounts.reset_index()
sector_wise_amounts = sector_wise_amounts.T.squeeze()

plt.figure(figsize=(12,8))
sns.barplot(y=sector_wise_amounts.index, x=sector_wise_amounts.values, alpha=0.6)
plt.ylabel("Sectors")
plt.xlabel("Funded Amounts")
plt.title("Distribution of Funded Amounts by Sectors in Business Type B")
df_business_classA[['funded_amount','loan_amount']].describe().T
df_business_classB[['funded_amount','loan_amount']].describe().T
classA_countries = list(df_business_classA['country'].unique()) 
classB_countries = list(df_business_classB['country'].unique())

temp = [i for i in classA_countries if i not in classB_countries] 
print ("Countries that are in classA type sectors(850+ USD funded loans) and are not in class B type sectors(funded loans<850 USDs) \n")
for i in temp : 
    print(i)
plt.figure(figsize=(12,10))
plt.scatter(df_business_classA['lender_count'],df_business_classA['funded_amount'])
plt.xlabel("Lender Count")
plt.ylabel("Funded Amount")
plt.title("Sector Class A - Scatter plot of Lender Count vs Funded Amount")
plt.figure(figsize=(12,10))
plt.scatter(df_business_classB['lender_count'],df_business_classB['funded_amount'])
plt.xlabel("Lender Count")
plt.ylabel("Funded Amount")
plt.title("Sector Class B - Scatter plot of Lender Count vs Funded Amount")
df_business_classA['borrower_genders'] = df_business_classA['borrower_genders'].str.split(",")

df_business_classB['borrower_genders'] = df_business_classB['borrower_genders'].str.split(",")
df_business_classA['borrower_count'] = df_business_classA['borrower_genders'].str.len()
df_business_classB['borrower_count'] = df_business_classB['borrower_genders'].str.len()
df_business_classA.head(2)
