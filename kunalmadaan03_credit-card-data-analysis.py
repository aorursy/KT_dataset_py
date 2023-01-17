import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
cust_acq = pd.read_csv("../input/creditcard/Customer Acqusition.csv")

cust_repay = pd.read_csv("../input/creditcard/Repayment.csv")

cust_spend = pd.read_csv("../input/creditcard/spend.csv")
cust_acq.head()
cust_repay.head()
cust_spend.head()
cust_spend=cust_spend.drop(["Sl No:"],axis=1)

cust_spend.head()
cust_repay=cust_repay.drop(["Unnamed: 4","SL No:"],axis=1)

cust_repay.dropna(inplace=True)

cust_repay.head()
cust_acq.Age = np.where(cust_acq.Age<18, round(cust_acq.Age.mean()),cust_acq.Age)
#Creating a new dataframe by joining cust_acq & cust_spend on basis of Customer column

cust_spend1 = pd.merge(left=cust_acq,right=cust_spend,on="Customer",how="inner")
cust_spend1.head()
# Find number of observations where Spend amount is > then customer limit

cust_spend1.loc[cust_spend1["Amount"] > cust_spend1["Limit"]]
cust_spend1.loc[cust_spend1["Amount"] > cust_spend1["Limit"],"Amount"] = (50 * cust_spend1["Limit"]/100)
cust_spend1.loc[cust_spend1["Amount"] > cust_spend1["Limit"]]
#Creating a new dataframe by joining cust_acq & cust_repay on basis of Customer column

cust_repay1 = pd.merge(left=cust_repay,right=cust_acq,on="Customer",how="inner")
cust_repay1.head()
# Find number of observations where Repay amount is > then customer limit

cust_repay1[cust_repay1.Amount > cust_repay1.Limit]
cust_repay1.loc[cust_repay1.Amount > cust_repay1.Limit,"Amount"] = cust_repay1.Limit
cust_repay1[cust_repay1.Amount > cust_repay1.Limit]
distinct_cust = cust_acq.Customer.nunique()
print("Total number of distinct customers are: ",distinct_cust)
dist_cat = cust_acq.Segment.value_counts()
print("Total distinct categories are: ",len(dist_cat),"\n Which are :\n",dist_cat)
# First convert Month in cust_spend table to datetime object

cust_spend.Month = pd.to_datetime(cust_spend.Month)

cust_spend.info()
# New month and year column

cust_spend['Monthly'] = cust_spend['Month'].apply(lambda x:pd.Timestamp.strftime(x,format="%b"))

cust_spend['Yearly'] = cust_spend['Month'].apply(lambda x:pd.Timestamp.strftime(x,format="%Y"))
cust_spend.head()
# Using groupby function average monthly spend by customers for respective years

out1 = cust_spend.groupby(["Yearly","Monthly"]).mean()
round(out1,2)
# First convert Month in cust_repay table to datetime object

cust_repay.Month = pd.to_datetime(cust_repay.Month)

cust_repay.info()
# New month and year column

cust_repay['Monthly'] = cust_repay['Month'].apply(lambda x:pd.Timestamp.strftime(x,format="%b"))

cust_repay['Yearly'] = cust_repay['Month'].apply(lambda x:pd.Timestamp.strftime(x,format="%Y"))
cust_repay.head()
# Using groupby function average monthly repayment done by customers for respective years

out2 = cust_repay.groupby(["Yearly","Monthly"]).mean()

round(out2,2)
# Merging all tables

# 1. cust_spend1 is already merged table of cust_acq and cust_spend

# 2. So merging cust_repay with cust_spend1



final_cust = pd.merge(left=cust_spend1,right=cust_repay,on="Customer",how="inner")
final_cust.head(1)
# Renaming the columns of final_cust tabe

final_cust.rename(columns={"Amount_x":"Spend_Amount","Amount_y":"Repay_Amount"},inplace=True)
final_cust.head(1)
interest_amt = final_cust.groupby(["Yearly","Monthly"])["Spend_Amount","Repay_Amount"].sum()
interest_amt
interest_amt["M_Profit"] = interest_amt.Repay_Amount - interest_amt.Spend_Amount
interest_amt
# As all profit amount is in positive.

interest_amt["M_interest"] = interest_amt.M_Profit * 2.9/100
interest_amt
top_products = cust_spend.Type.value_counts().head(5)

top_products
city_spend = cust_spend1.groupby("City")["Amount"].sum()

print(city_spend.idxmax()," has max spend.")
# New column "age_group" with 8 bins between 18 to 88 

cust_spend1["age_group"] =  pd.cut(cust_spend1["Age"],bins=np.arange(18,88,8),labels=["18-26","26-34", "34-42" ,"42-50" ,"50-58","58-66","66-74","74-82"],include_lowest=True)

cust_spend1.head(2)
age_spend = cust_spend1.groupby("age_group")["Amount"].sum()

print("Age group (",age_spend.idxmax(),") has max spend.")
cust_repay1.groupby("Customer")[["Amount"]].sum().sort_values(by="Amount",ascending=False).head(10)
# Converting "Month" column to date time 

cust_spend1["Month"] = pd.to_datetime(cust_spend1["Month"])
# Creating new column "year" 

cust_spend1['Year'] = cust_spend1['Month'].apply(lambda x:pd.Timestamp.strftime(x,format="%Y"))

cust_spend1.head(2)
out3 = pd.pivot_table(data = cust_spend1,index=["City","Year"],columns='Product',aggfunc="sum",values="Amount")
out3
out3.plot(kind="bar",figsize=(18,5),width=0.8)

plt.ylabel("Spend Amount")

plt.title("Amount spended by customers according to year and city")

plt.show()
# Creating column "Monthly" for graph 

cust_spend1['Monthly'] = cust_spend1['Month'].apply(lambda x:pd.Timestamp.strftime(x,format="%B"))
out1a = cust_spend1.groupby(["Monthly","City"])[["Amount"]].sum().sort_index().reset_index()

out1a =pd.pivot_table(data=cust_spend1,values='Amount',index='City',columns='Monthly',aggfunc='sum')

out1a
out1a.plot(kind="bar",figsize=(20,7),width=0.8)

plt.show()
spend_tickets = cust_spend1.groupby(["Year","Type"])[["Amount"]].sum().reset_index()
out1b = spend_tickets.loc[spend_tickets["Type"]=="AIR TICKET"]

out1b
plt.bar(out1b["Year"],height=out1b["Amount"],color="blue")

plt.xlabel("Year")

plt.ylabel("Amount Spent")

plt.title("Spend on air tickets")

plt.show()
out2a = pd.pivot_table(data=cust_spend1,index='Product',columns='Monthly',values='Amount',aggfunc='sum')
out2a
out2a.plot(kind="bar",figsize=(18,6),width=0.8)

plt.show()
cust_repay1['Month'] = pd.to_datetime(cust_repay1['Month'])

#creating new column "Monthly" and "Yearly" using already existing 'Month' column

cust_repay1['Monthly'] = cust_repay1['Month'].apply(lambda x:pd.Timestamp.strftime(x,format="%B"))

cust_repay1['Yearly'] = cust_repay1['Month'].apply(lambda x:pd.Timestamp.strftime(x,format="%Y"))
def top_cust(product,time_period):

    print('Give the product name and time_period for which you want the data')

    if product.lower()=='gold' and time_period.lower()=='monthly':

        table3 = cust_repay1.pivot_table(index=['Product','City','Customer'],columns='Monthly',aggfunc='sum',values='Amount')

        result = table3.loc[('Gold',['BANGALORE','COCHIN','CALCUTTA','BOMBAY','CHENNAI','TRIVANDRUM','PATNA','DELHI']),:]

    elif product.lower()=='gold' and time_period.lower()=='yearly':

        table3 = cust_repay1.pivot_table(index=['Product','City','Customer'],columns='Yearly',aggfunc='sum',values='Amount')

        result = table3.loc[('Gold',['BANGALORE','COCHIN','CALCUTTA','BOMBAY','CHENNAI','TRIVANDRUM','PATNA','DELHI']),:]

    elif product.lower()=='silver' and time_period.lower()=='monthly':

        table3 = cust_repay1.pivot_table(index=['Product','City','Customer'],columns='Monthly',aggfunc='sum',values='Amount')

        result = table3.loc[('Silver',['BANGALORE','COCHIN','CALCUTTA','BOMBAY','CHENNAI','TRIVANDRUM','PATNA','DELHI']),:]

    elif product.lower()=='silver' and time_period.lower()=='yearly':

        table3 = cust_repay1.pivot_table(index=['Product','City','Customer'],columns='Yearly',aggfunc='sum',values='Amount')

        result = table3.loc[('Silver',['BANGALORE','COCHIN','CALCUTTA','BOMBAY','CHENNAI','TRIVANDRUM','PATNA','DELHI']),:]

    if product.lower()=='platimum' and time_period.lower()=='monthly':

        table3 = cust_repay1.pivot_table(index=['Product','City','Customer'],columns='Monthly',aggfunc='sum',values='Amount')

        result = table3.loc[('Platimum',['BANGALORE','COCHIN','CALCUTTA','BOMBAY','CHENNAI','TRIVANDRUM','PATNA','DELHI']),:]

    elif product.lower()=='platimum' and time_period.lower()=='yearly':

        table3 = cust_repay1.pivot_table(index=['Product','City','Customer'],columns='Yearly',aggfunc='sum',values='Amount')

        result = table3.loc[('Platimum',['BANGALORE','COCHIN','CALCUTTA','BOMBAY','CHENNAI','TRIVANDRUM','PATNA','DELHI']),:]

    return result
top_cust("platimum","yearly")