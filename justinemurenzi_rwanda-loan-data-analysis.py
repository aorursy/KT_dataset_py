import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
kiva=pd.read_csv('kiva_loans.csv')
rwanda_df= kiva[kiva['country']=='Rwanda']
plt.figure(figsize=(10,5))
sns.barplot(x='sector',y='loan_amount',data=rwanda_df )
plt.xlabel('Sector',weight='bold',fontsize=14)
plt.ylabel('Loan Amount',weight='bold',fontsize=14)
plt.title('Loan Application per sector In RWANDA', weight='bold',fontsize=14)
plt.xticks(rotation=90)
plt.grid(True)
hue_colors={'male':'black','female': 'red'}
plt.figure(figsize=(5,5))
sns.catplot(x='borrower_genders',y='loan_amount',kind='bar',data=rwanda_df,palette=hue_colors)
plt.title('Loan Borrower by Gender',weight='bold',fontsize=14)
plt.xlabel('Gender',weight='bold',fontsize=14)
plt.ylabel('Loan Amount (RFW)',weight='bold',fontsize=14)
sns.catplot(x="repayment_interval",y="loan_amount", data=rwanda_df,kind="bar",ci=None,hue="borrower_genders",palette=hue_colors)
plt.title("Payment Interval Comparision",weight="bold", fontsize=14)
plt.xlabel("Payment Schedule",weight="bold",fontsize=14)
plt.ylabel("Loan Amount",weight="bold",fontsize=14)
sns.scatterplot(x="loan_amount",y="lender_count",data=rwanda_df,hue="borrower_genders", sizes=(20, 200),palette=hue_colors)
plt.xlabel("Loan Amount in (RFW)",weight="bold",fontsize=16)
plt.ylabel("Lender Count",weight="bold",fontsize=16)
plt.title("Lender Counts against Loan Value", weight='bold',fontsize=16)
plt.show()
sns.scatterplot(x="term_in_months",y="loan_amount",data=rwanda_df,hue="borrower_genders",palette=hue_colors)
plt.xlabel("term_in_months",weight="bold",fontsize=16)
plt.ylabel("Loan Amount",weight="bold",fontsize=16)
plt.title("Loan Term against Loan Amount", weight='bold',fontsize=16)
plt.show()
