import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#kiva_df=pd.read_csv('/kaggle/input/kiva_loans.csv')
kiva_df= pd.read_csv("/kaggle/input/kiva-loans-data/kiva_loans.csv")
us_df=kiva_df[kiva_df["country_code"]=="US"]
plt.figure(figsize=(10,5))
sns.barplot(x='sector',y='loan_amount',data=us_df,ci=None)
plt.xticks(rotation=90)
plt.title("Loan Application per Sector", weight='bold')
plt.xlabel("Sector",weight="bold",fontsize=14)
plt.ylabel("Loan Amount", weight="bold",fontsize=14)
plt.xticks(rotation=90)
plt.grid(True)
#Defination of the Colors using a dictionary
hue_colors={"male":"blue","female":"magenta"}

#Plot the bar chart
plt.figure(figsize=(4,4))
sns.catplot(x="borrower_genders",y="loan_amount",kind='bar',data=us_df,ci=None,palette=hue_colors)
plt.title("Loan Borrowed by Gender",weight="bold")
plt.xlabel("Gender",weight="bold")
plt.ylabel("Loan Amount ( USD)")
plt.show()
plt.figure(figsize=(10,5))

sns.catplot(x="repayment_interval",y="loan_amount", data=us_df,kind="bar",ci=None,hue="borrower_genders",palette=hue_colors)
plt.title("Payment Interval Comparision",weight="bold")
plt.xlabel("Payment Schedule",weight="bold")
plt.ylabel("Loan Amount",weight="bold")
plt.figure(figsize=(10,5))

sns.scatterplot(x="loan_amount",y="lender_count",data=us_df,hue="borrower_genders", sizes=(20, 200),palette=hue_colors)
plt.xlabel("Loan Amount in (dollars)",weight="bold")
plt.ylabel("Lender Count",weight="bold")
plt.title("Lender Counts against Loan Value", weight='bold')
plt.show()
plt.figure(figsize=(10,5))

sns.scatterplot(x="term_in_months",y="loan_amount",data=us_df,hue="borrower_genders",palette=hue_colors)
plt.xlabel("term_in_months",weight="bold")
plt.ylabel("Loan Amount",weight="bold")
plt.title("Loan Term against Loan Amount", weight='bold')
plt.show()
