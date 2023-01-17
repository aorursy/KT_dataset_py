# Data Manipulation libraries
import pandas as pd
import numpy as np
import missingno as msno

from datetime import timedelta

# Data Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# Loading the data
data = pd.read_csv("../input/ecommerce-data/data.csv", encoding="unicode_escape", parse_dates=['InvoiceDate'])
print("Total number of transactions happened in the given period: "+ str(data.shape[0]))
data.head(5)
# Checking the data types
data.info()
# Checking for missing Values
pd.DataFrame(zip(data.isnull().sum(), data.isnull().sum()/len(data)), columns=['count', 'proportion'], index=data.columns)
# Visualizing the missing values
msno.matrix(data)
# Removing null CustomerIDs
main = data.copy()
data = data[pd.notnull(data['CustomerID'])]
# Descriptive Statistics
data.describe()
# Filter for most frequently returned items
refunds = data[data['Quantity']<0]
refunds = refunds.groupby('Description')['Quantity'].count().sort_values(ascending=False).reset_index()[:10]

plt.figure(figsize=(16,4))
sns.barplot(x= refunds['Description'], y=refunds['Quantity'])
plt.title("Top 10 more frequently returned items", size=15)
plt.xticks(rotation=90)
plt.xlabel(" ")
plt.show()
data[data['UnitPrice']>1000].tail(7)
data = data[~data['Description'].isin(['POSTAGE', 'DOTCOM POSTAGE', 'CRUK Commission', 'Manual'])]
data.describe()
data['TotalSales'] = data['Quantity'].multiply(data['UnitPrice'])
data.head()
# Printing the details of the dataset
maxdate = data['InvoiceDate'].dt.date.max()
mindate = data['InvoiceDate'].dt.date.min()
unique_cust = data['CustomerID'].nunique()
unique_stocks = data['StockCode'].nunique()
tot_quantity = data['Quantity'].sum()
tot_sales = data['TotalSales'].sum()

print(f"The Time range of transactions is: {mindate} to {maxdate}")
print(f"Total number of unique customers: {unique_cust}")
print(f"Total number of unique stocks: {unique_stocks}")
print(f"Total Quantity Sold: {tot_quantity}")
print(f"Total Sales for the period: {tot_sales}")
# The most purchased products from the website
top_purchase = data.groupby('Description')['TotalSales'].count().sort_values(ascending=False)[:10]

plt.figure(figsize=(16,4))
sns.barplot(x=top_purchase.index, y=top_purchase.values)
plt.xticks(rotation=90)
plt.title("Top 10 Most Purchased Products", size=15)
plt.xlabel(" ")
plt.show()
# Top countries by sales value
top_country = data.groupby('Country')['TotalSales'].sum().sort_values(ascending=False)[:10]

labels = top_country[:5].index
size = top_country[:5].values

plt.figure(figsize=(7,7))
plt.pie(size, labels=labels, explode=[0.05]*5, autopct='%1.0f%%')
plt.title("Top 5 Countries by Total Sales Value", size=15)
plt.axis('equal')
plt.show()
data['Hour'] = data['InvoiceDate'].dt.hour
data['Weekday'] = data['InvoiceDate'].dt.weekday
data['Month'] = data['InvoiceDate'].dt.month
data.head(3)
# Transaction trend in hour of the day
hour = data.groupby('Hour')['Quantity'].count()
plt.figure(figsize=(16,4))
hour.plot(marker='+')
plt.title("Transactions - Hourly", size=15)
plt.show()
# Transaction trend in day of week
weekday = data.groupby('Weekday')['Quantity'].count()
plt.figure(figsize=(16,4))
weekday.plot(marker='^')
plt.title("Transactions - Day of Week", size=15)
plt.show()
# Transaction trend across months
month = data.groupby('Month')['Quantity'].count()
plt.figure(figsize=(16,4))
month.plot(marker='+')
plt.title("Transactions - Monthly", size=15)
plt.show()
def RFM_Features(df, customerID, invoiceDate, transID, sales):
    ''' Create the Recency, Frequency, and Monetary features from the data '''
    # Final date in the data + 1 to create latest date
    latest_date = df[invoiceDate].max() + timedelta(1)
    
    # RFM feature creation
    RFMScores = df.groupby(customerID).agg({invoiceDate: lambda x: (latest_date - x.max()).days, 
                                          transID: lambda x: len(x), 
                                          sales: lambda x: sum(x)})
    
    # Converting invoiceDate to int since this contains number of days
    RFMScores[invoiceDate] = RFMScores[invoiceDate].astype(int)
    
    # Renaming column names to Recency, Frequency and Monetary
    RFMScores.rename(columns={invoiceDate: 'Recency', 
                         transID: 'Frequency', 
                         sales: 'Monetary'}, inplace=True)
    
    return RFMScores.reset_index()
RFM = RFM_Features(df=data, customerID= "CustomerID", invoiceDate = "InvoiceDate", transID= "InvoiceNo", sales="TotalSales")
RFM.head()
# Descriptive Stats
display(RFM.describe())

# Distributions of Recency, Frequency, and Monetary features
# Here we will filter out the extreme values in the Frequency and Monetary columns to avoid the skewness the distribution
fig, ax = plt.subplots(1,3, figsize=(14,6))

sns.distplot(RFM.Recency, bins=20, ax=ax[0])
sns.distplot(RFM[RFM['Frequency']<1000]['Frequency'], bins=20, ax=ax[1])
sns.distplot(RFM[RFM['Monetary']<10000]['Monetary'], bins=20, ax=ax[2])
plt.show()
# Creating quantiles 
Quantiles = RFM[['Recency', 'Frequency', 'Monetary']].quantile([0.25, 0.50, 0.75])
Quantiles = Quantiles.to_dict()
Quantiles
# Creating RFM ranks
def RFMRanking(x, variable, quantile_dict):
    ''' Ranking the Recency, Frequency, and Monetary features based on quantile values '''
    
    # checking if the feature to rank is Recency
    if variable == 'Recency':
        if x <= quantile_dict[variable][0.25]:
            return 4
        elif (x > quantile_dict[variable][0.25]) & (x <= quantile_dict[variable][0.5]):
            return 3
        elif (x > quantile_dict[variable][0.5]) & (x <= quantile_dict[variable][0.75]):
            return 2
        else:
            return 1
    
    # checking if the feature to rank is Frequency and Monetary
    if variable in ('Frequency','Monetary'):
        if x <= quantile_dict[variable][0.25]:
            return 1
        elif (x > quantile_dict[variable][0.25]) & (x <= quantile_dict[variable][0.5]):
            return 2
        elif (x > quantile_dict[variable][0.5]) & (x <= quantile_dict[variable][0.75]):
            return 3
        else:
            return 4
RFM['R'] = RFM['Recency'].apply(lambda x: RFMRanking(x, variable='Recency', quantile_dict=Quantiles))
RFM['F'] = RFM['Frequency'].apply(lambda x: RFMRanking(x, variable='Frequency', quantile_dict=Quantiles))
RFM['M'] = RFM['Monetary'].apply(lambda x: RFMRanking(x, variable='Monetary', quantile_dict=Quantiles))
RFM.head()
RFM['Group'] = RFM['R'].apply(str) + RFM['F'].apply(str) + RFM['M'].apply(str)
RFM.head()
# Check the number of score segments
RFM.Group.value_counts()
RFM["Score"] = RFM[['R', 'F', 'M']].sum(axis=1)
RFM.head()
# Loyalty levels
loyalty = ['Bronze', 'Silver', 'Gold', 'Platinum']
RFM['Loyalty_Level'] = pd.qcut(RFM['Score'], q=4, labels= loyalty)
RFM.head()
behaviour = RFM.groupby('Loyalty_Level')[['Recency', 'Frequency', 'Monetary', 'Score']].mean()
behaviour