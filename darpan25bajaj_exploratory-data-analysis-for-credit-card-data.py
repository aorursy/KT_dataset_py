import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import os


for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
customer = pd.read_csv("/kaggle/input/credit-card-exploratory-data-analysis/Customer Acqusition.csv",usecols=["Customer","Age","City","Product","Limit","Company","Segment"])

repay = pd.read_csv("/kaggle/input/credit-card-exploratory-data-analysis/Repayment.csv",usecols = ["Customer","Month","Amount"])

spend = pd.read_csv("/kaggle/input/credit-card-exploratory-data-analysis/spend.csv",usecols=["Customer","Month","Type","Amount"])
customer.head()
repay.head()
spend.head(2)
print(customer.shape)

print(repay.shape)

print(spend.shape)
customer.dtypes
repay.dtypes
spend.dtypes
spend.isnull().sum()
customer.isnull().sum()
repay.isnull().sum()
# dropping null values present in 'repay' data set

repay.dropna(inplace=True)
repay.isnull().sum()
mean_original = customer["Age"].mean()
print("The mean of Age column is",mean_original)
#replacing age less than 18 with mean of age values

customer.loc[customer["Age"] < 18,"Age"] = customer["Age"].mean()
mean_new = customer["Age"].mean()
print("The new mean of Age column is",mean_new)
customer.loc[customer["Age"] < 18,"Age"]
print("All the customers who have age less than 18 have been replaced by mean of the age column.")
customer.head(2)
spend.head(2)
#merging customer and spend table on the basis of "Customer" column

customer_spend = pd.merge(left=customer,right=spend,on="Customer",how="inner")
customer_spend.head()
customer_spend.shape
#all the customers whose spend amount is more than the limit,replacing with 50% of that customer’s limit

customer_spend[customer_spend["Amount"] > customer_spend['Limit']]
#if customer's spend amount is more than the limit,replacing with 50% of that customer’s limit

customer_spend.loc[customer_spend["Amount"] > customer_spend["Limit"],"Amount"] = (50 * customer_spend["Limit"]).div(100)
#there are no customers left whose spend amount is more than the limit

customer_spend[customer_spend["Amount"] > customer_spend['Limit']]
customer.head(1)
repay.head(1)
#merging customer and spend table on the basis of "Customer" column

customer_repay = pd.merge(left=repay,right=customer,on="Customer",how="inner")
customer_repay.head()
#all the customers where repayment amount is more than the limit.

customer_repay[customer_repay["Amount"] > customer_repay["Limit"]]
#customers where repayment amount is more than the limit, replacing the repayment with the limit.

customer_repay.loc[customer_repay["Amount"] > customer_repay["Limit"],"Amount"] = customer_repay["Limit"]
#there are no customers left where repayment amount is more than the limit.

customer_repay[customer_repay["Amount"] > customer_repay["Limit"]]
distinct_customers = customer["Customer"].nunique()
print("Number of distinct customers are",distinct_customers)
#customers from different segments

customer["Segment"].value_counts()
plt.figure(figsize=(8,6))

sns.countplot('Segment',data=customer)

plt.show()
print("We can see from the countplot that number of distinct categories are", len(customer["Segment"].value_counts()))
spend.head()
#converting Month column of "spend" table to date time format

spend['Month'] = pd.to_datetime(spend['Month'])
spend.head()
#creating new columns which show "Month" and "Year"

spend['Monthly'] = spend['Month'].apply(lambda x:pd.Timestamp.strftime(x,format="%B"))

spend['Yearly'] = spend['Month'].apply(lambda x:pd.Timestamp.strftime(x,format="%Y"))
spend.head()
#grouping the dataset based on 'Yearly' and 'monthly'

customer_spend_group= round(spend.groupby(['Yearly','Monthly']).mean(),2)
customer_spend_group
repay.head(2)
#coverting "Month" column to date time format

repay["Month"] = pd.to_datetime(repay["Month"])
repay.head(2)
repay.dtypes
#creating new columns which show "Month" and "Year"

repay['Monthly'] = repay['Month'].apply(lambda x:pd.Timestamp.strftime(x,format="%B"))

repay['Yearly'] = repay['Month'].apply(lambda x:pd.Timestamp.strftime(x,format="%Y"))
#grouping the dataset based on 'Yearly' and 'monthly'

customer_repay_group= round(repay.groupby(['Yearly','Monthly']).mean(),2)
customer_repay_group
#merging all the three tables. Alreaady merged customer and spend table in 'customer_spend'. Using "customer_spend" and "repay"

#table to form the final "customer_spend_repay" table

customer_spend_repay = pd.merge(left=customer_spend,right=repay,on="Customer",how="inner")
customer_spend_repay.head(2)
# renaming the columns for clearity

customer_spend_repay.rename(columns={"Amount_x":"Spend_Amount","Amount_y":"Repay_Amount"},inplace=True)
customer_spend_repay.head()
# grouping the data based on "Yearly","Month_x" columns to get the 'Spend_Amount'and 'Repay_Amount'

interest_group = customer_spend_repay.groupby(["Yearly","Monthly"])['Spend_Amount','Repay_Amount'].sum()
interest_group
 # Monthly Profit = Monthly repayment – Monthly spend.

interest_group['Monthly Profit'] = interest_group['Repay_Amount'] - interest_group['Spend_Amount']
interest_group
#interest earned is 2.9% of Monthly Profit

interest_group['Interest Earned'] = (2.9* interest_group['Monthly Profit'])/100
interest_group
spend.head()
#top 5 product types on which customer is spending

spend['Type'].value_counts().head()
spend['Type'].value_counts().head(5).plot(kind='bar')

plt.show()
customer_spend.head()
city_spend = customer_spend.groupby("City")["Amount"].sum().sort_values(ascending=False)
city_spend
plt.figure(figsize=(5,10))

city_spend.plot(kind="pie",autopct="%1.0f%%",shadow=True,labeldistance=1.0,explode=[0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0])

plt.title("Amount spent on credit card by customers from different cities")

plt.show()
print("From above pie chart we can see that Cochin is having maximum spend.")
#creating new column "Age Group" with 8 bins between 18 to 88 

customer_spend["Age Group"] =  pd.cut(customer_spend["Age"],bins=np.arange(18,88,8),labels=["18-26","26-34", "34-42" ,"42-50" ,"50-58","58-66","66-74","74-82"],include_lowest=True)
customer_spend.head()
#grouping data based on "Age Group" and finding the amount spend by each age group and arranging in descending oreder

age_spend = customer_spend.groupby("Age Group")['Amount'].sum().sort_values(ascending=False)
age_spend
plt.figure(figsize=(5,10))

age_spend.plot(kind = "pie",autopct="%1.0f%%",explode=[0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0],shadow=True)

plt.show()
print("From the pie chart shown above we can say that age group 42 - 50 is spending more money")
customer_repay.head()
#grouping based on "Customer" column to find top 10 customers

customer_repay.groupby("Customer")[["Amount"]].sum().sort_values(by="Amount",ascending=False).head(10)
customer_spend.head()
#converting "Month" column to date time 

customer_spend["Month"] = pd.to_datetime(customer_spend["Month"])
#creating new column "year" 

customer_spend['Year'] = customer_spend['Month'].apply(lambda x:pd.Timestamp.strftime(x,format="%Y"))
customer_spend.head(2)
customer_spend_pivot = pd.pivot_table(data = customer_spend,index=["City","Year"],columns='Product',aggfunc="sum",values="Amount")
customer_spend_pivot
customer_spend_pivot.plot(kind="bar",figsize=(18,5),width=0.8)

plt.ylabel("Spend Amount")

plt.title("Amount spended by customers according to year and city")

plt.show()
customer_spend.head()
#creating new column "Monthly" 

customer_spend['Monthly'] = customer_spend['Month'].apply(lambda x:pd.Timestamp.strftime(x,format="%B"))
customer_spend.head()
#grouping data based on "Monthly" and "City" columns

month_city = customer_spend.groupby(["Monthly","City"])[["Amount"]].sum().sort_index().reset_index()
#creating pivot table based on "Monthly" and "City" columns

month_city =pd.pivot_table(data=customer_spend,values='Amount',index='City',columns='Monthly',aggfunc='sum')
month_city
month_city.plot(kind="bar",figsize=(18,6),width=0.8)

plt.show()
customer_spend.head()
air_tickets = customer_spend.groupby(["Year","Type"])[["Amount"]].sum().reset_index()
filtered = air_tickets.loc[air_tickets["Type"]=="AIR TICKET"]
filtered
plt.bar(filtered["Year"],height=filtered["Amount"],color="orange")

plt.xlabel("Year")

plt.ylabel("Amount Spent")

plt.title("Comparison of yearly spend on air tickets")

plt.show()

customer_spend.head(2)
#creating pivot table based on "Monthly" and "Product" columns

product_wise = pd.pivot_table(data=customer_spend,index='Product',columns='Monthly',values='Amount',aggfunc='sum')
product_wise
product_wise.plot(kind="bar",figsize=(18,6),width=0.8)

plt.ylabel("Amount Spend")

plt.title("Amount spent monthly on different products")

plt.show()
customer_repay.head(2)
# converting 'Month' column to date time format

customer_repay['Month'] = pd.to_datetime(customer_repay['Month'])
#creating new column "Monthly" and "Yearly" using already existing 'Month' column

customer_repay['Monthly'] = customer_repay['Month'].apply(lambda x:pd.Timestamp.strftime(x,format="%B"))

customer_repay['Yearly'] = customer_repay['Month'].apply(lambda x:pd.Timestamp.strftime(x,format="%Y"))
def summary_report(product,timeperiod):

    print('Give the product name and timeperiod for which you want the data')

    if product.lower()=='gold' and timeperiod.lower()=='monthly':

        pivot = customer_repay.pivot_table(index=['Product','City','Customer'],columns='Monthly',aggfunc='sum',values='Amount')

        result = pivot.loc[('Gold',['BANGALORE','COCHIN','CALCUTTA','BOMBAY','CHENNAI','TRIVANDRUM','PATNA','DELHI']),:]

    elif product.lower()=='gold' and timeperiod.lower()=='yearly':

        pivot = customer_repay.pivot_table(index=['Product','City','Customer'],columns='Yearly',aggfunc='sum',values='Amount')

        result = pivot.loc[('Gold',['BANGALORE','COCHIN','CALCUTTA','BOMBAY','CHENNAI','TRIVANDRUM','PATNA','DELHI']),:]

    elif product.lower()=='silver' and timeperiod.lower()=='monthly':

        pivot = customer_repay.pivot_table(index=['Product','City','Customer'],columns='Monthly',aggfunc='sum',values='Amount')

        result = pivot.loc[('Silver',['BANGALORE','COCHIN','CALCUTTA','BOMBAY','CHENNAI','TRIVANDRUM','PATNA','DELHI']),:]

    elif product.lower()=='silver' and timeperiod.lower()=='yearly':

        pivot = customer_repay.pivot_table(index=['Product','City','Customer'],columns='Yearly',aggfunc='sum',values='Amount')

        result = pivot.loc[('Silver',['BANGALORE','COCHIN','CALCUTTA','BOMBAY','CHENNAI','TRIVANDRUM','PATNA','DELHI']),:]

    if product.lower()=='platinum' and timeperiod.lower()=='monthly':

        pivot = customer_repay.pivot_table(index=['Product','City','Customer'],columns='Monthly',aggfunc='sum',values='Amount')

        result = pivot.loc[('Platinum',['BANGALORE','COCHIN','CALCUTTA','BOMBAY','CHENNAI','TRIVANDRUM','PATNA','DELHI']),:]

    elif product.lower()=='platinum' and timeperiod.lower()=='yearly':

        pivot = customer_repay.pivot_table(index=['Product','City','Customer'],columns='Yearly',aggfunc='sum',values='Amount')

        result = pivot.loc[('Platinum',['BANGALORE','COCHIN','CALCUTTA','BOMBAY','CHENNAI','TRIVANDRUM','PATNA','DELHI']),:]

    return result
summary_report('gold','monthly')