import numpy as np   #numpy for matehmatical computations
import pandas as pd  #pandas for data computations


import warnings      #warnings to supress second and subsequent repeated warnings
warnings.filterwarnings("ignore")

#datetime imports for date - time computations
import time
import datetime as dt
from datetime import date
from datetime import datetime
from datetime import timedelta


import json #json for handling json data

#Visualization Imports
import seaborn as sns
sns.set_palette('Pastel2')
import matplotlib.pyplot as plt
%matplotlib inline
from plotly import __version__
#%matplotlib inline

from plotly.tools import FigureFactory as FF
import cufflinks as cf
import plotly as py
import plotly.offline as pyo
import plotly.graph_objs as go
import folium
import plotly.express as px
import plotly.tools as tls
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
cf.go_offline()

#StandardScaler Data Normalization Imports
from sklearn.preprocessing import StandardScaler

#Kmeans cluter algorithm import
from sklearn.cluster import KMeans

#Identify Silhouette score for best KMeans cluster import
from sklearn.metrics import silhouette_score

# Supress Scientific notation in python import
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# Display all columns of long dataframe
pd.set_option('display.max_columns', None)
path = "../input/input-datasets/"
# Import input datasets
train = pd.read_excel(path+'train.xlsx', parse_dates=['InvoiceDate'])
test = pd.read_excel(path+'test.xlsx', parse_dates=['InvoiceDate'])
#Bonus Steps : Import Country Dataset - For Visualization in Tableau have added a Country to Country Code Mapping
countries = pd.read_excel(path+'Country.xlsx')
# Check the shape of dataframe
train.shape
train.head()
# Check the shape of dataframe
test.shape
test.head()
data = pd.concat([train, test], ignore_index=True)
data.reset_index(drop=True,inplace = True)
data.shape
data.head()
data.info()
data.describe()
dataDuplicates = pd.DataFrame(columns=['Type','Count'])
dataDuplicates = dataDuplicates.append({'Type': "Duplicates",'Count' : data.duplicated().sum()}, ignore_index=True)
dataDuplicates = dataDuplicates.append({'Type': "Unique Value",'Count' : data.count()[0] - data.duplicated().sum()}, ignore_index=True)
dataDuplicates
# Set notebook mode to work in offline
pyo.init_notebook_mode()
# Use `hole` to create a donut-like pie chart
fig = go.Figure(data=[go.Pie(labels=dataDuplicates.Type, values=dataDuplicates.Count, title = "Duplicates Spread",hole=.8)])
fig.show()
data.drop_duplicates(inplace=True)
data.reset_index(drop=True,inplace = True)
data.shape
data.CustomerID.notnull().sum()
dataCustomerID = pd.DataFrame(columns=['Type','Count'])
dataCustomerID = dataCustomerID.append({'Type': "Null Values",'Count' : data.count()[0] - data.CustomerID.notnull().sum()}, 
                                       ignore_index=True)
dataCustomerID = dataCustomerID.append({'Type': "Populated Values",'Count' : data.CustomerID.notnull().sum()}, ignore_index=True)
dataCustomerID
# Set notebook mode to work in offline
pyo.init_notebook_mode()
# Use `hole` to create a donut-like pie chart
fig = go.Figure(data=[go.Pie(labels=dataCustomerID.Type, values=dataCustomerID.Count, 
                             title = "Customer ID Missing Values",hole=.8)])
fig.show()
# We don't need records with Null customer id for RFM analysis so let's remove those first
data = data[data.CustomerID.notnull()]
data.reset_index(drop=True,inplace = True)
data.shape
data.info()
# title function in python to convert the character column to title case or proper case
 
data['Description'] = data['Description'].str.title()
data.head()
# Convert CustomerID to int type
data.CustomerID = (data.CustomerID).astype(int)
data.head(5)
# Convert CustomerId to str type
data.CustomerID = (data.CustomerID).astype(str)
data.info()
groupByCountry = data.groupby(["Country"], as_index=False)['InvoiceNo'].count().rename(columns={'Country':'Country','InvoiceNo' : 'Count'})
groupByCountry.head()
cf.set_config_file(theme='white')
groupByCountry.iplot(kind = 'bar', x = 'Country', y = 'Count', title = 'Transaction Count of Purchases by Country',
                     xTitle='Country', yTitle='#Purchased')
data.Country.value_counts(normalize=True).head(3).mul(100).round(1).astype(str) + '%'
# Create feature total cost of the transactions
data['TotalCost'] = data.Quantity * data.UnitPrice
data.head()
groupBySumTransactionCountry = data.groupby(['Country'], as_index=False)['TotalCost'].sum().rename(columns={'Country':'Country','TotalCost' : 'SumPurchase'})
groupBySumTransactionCountry = groupBySumTransactionCountry.sort_values(by='SumPurchase', ascending=False)
groupBySumTransactionCountry.head()
cf.set_config_file(theme='white')
groupBySumTransactionCountry.iplot(kind = 'bar', x = 'Country', y = 'SumPurchase', title = 'Sum Purchases By Country',
                     xTitle='Country', yTitle='Purchase Sum in £')
groupByAverageTransactionCountry = data.groupby(['Country'], as_index=False)['TotalCost'].mean().rename(columns={'Country':'Country','TotalCost' : 'MeanPurchase'})
groupByAverageTransactionCountry = groupByAverageTransactionCountry.sort_values(by='MeanPurchase', ascending=False)
groupByAverageTransactionCountry.head()
cf.set_config_file(theme='white')
groupByAverageTransactionCountry.iplot(kind = 'bar', x = 'Country', y = 'MeanPurchase', title = 'Mean Purchases By Country',
                     xTitle='Country', yTitle='Purchase Mean in £ Per Transaction')
# Count of transactions in different years
groupYear = data.InvoiceDate.dt.year.value_counts(sort=False).rename_axis('Year').reset_index(name='TransactionCount')
groupYear.Year = groupYear.Year.astype(str)
groupYear.Year = 'Year ' + groupYear.Year
cf.set_config_file(theme='white')
groupYear.iplot(kind = 'barh', x = 'Year', y = 'TransactionCount', title = 'Transactions By Year',
                     yTitle='Year of Sale', xTitle='# of Transactions',text = 'TransactionCount')
# Count of transactions in 2011
groupMonth = data[data.InvoiceDate.dt.year==2011].InvoiceDate.dt.month.value_counts(sort=False).rename_axis('Month').reset_index(name='TransactionCount')
groupMonth
groupMonth.Month = groupMonth.Month.astype(str) + '-2011'
groupMonth
cf.set_config_file(theme='white')
groupMonth.iplot(kind = 'barh', x = 'Month', y = 'TransactionCount', title = 'Transactions By Month in 2011',
                     yTitle='Month of Sale', xTitle='# of Transactions',text = 'TransactionCount')
groupByDescription = data.groupby(["Description"], as_index=False)['InvoiceNo'].count().rename(columns={'Description':'Description','InvoiceNo' : 'Count'})
groupByDescription = groupByDescription.sort_values(by='Count', ascending=False).head(20)
cf.set_config_file(theme='white')
groupByDescription.iplot(kind = 'barh', x = 'Description', y = 'Count', title = 'Top 20 Spread of Purchases by Description',
                     yTitle='Description', xTitle='#Purchased',text = 'Count')
groupByTotalCost = data.groupby(["InvoiceNo"], as_index=False)['TotalCost'].sum().rename(columns={'InvoiceNo':'InvoiceNo','TotalCost' : 'PurchaseTotal'})
groupByTotalCost.InvoiceNo = groupByTotalCost.InvoiceNo.astype(str)
groupByTotalCost = groupByTotalCost.sort_values(by='PurchaseTotal', ascending=False).head(20)
groupByTotalCost.InvoiceNo = "INV " + groupByTotalCost.InvoiceNo
groupByTotalCost.head()
cf.set_config_file(theme='white')
groupByTotalCost.iplot(kind = 'barh', x = 'InvoiceNo', y = 'PurchaseTotal', title = 'Top 20 Purchases by Invoice Number',
                     yTitle='Invoice Number', xTitle='Purchase Total')
groupByTotalCost2 = data.groupby(["InvoiceNo"], as_index=False)['TotalCost'].sum().rename(columns={'InvoiceNo':'InvoiceNo','TotalCost' : 'PurchaseTotal'})
groupByTotalCost2.InvoiceNo = groupByTotalCost2.InvoiceNo.astype(str)
groupByTotalCost2 = groupByTotalCost2.sort_values(by='PurchaseTotal', ascending=True).head(20)
groupByTotalCost2.InvoiceNo = "INV " + groupByTotalCost2.InvoiceNo
groupByTotalCost2.head()
cf.set_config_file(theme='white')
groupByTotalCost2.iplot(kind = 'barh', x = 'InvoiceNo', y = 'PurchaseTotal', title = 'Bottom 20 Purchases by Invoice Number',
                     yTitle='Invoice Number', xTitle='Purchase Total')
groupByTotalCost1 = data.groupby(["CustomerID"], as_index=False)['TotalCost'].sum().rename(columns={'CustomerID':'CustomerID','TotalCost' : 'PurchaseTotal'})
groupByTotalCost1.CustomerID = groupByTotalCost1.CustomerID.astype(str)
groupByTotalCost1 = groupByTotalCost1.sort_values(by='PurchaseTotal', ascending=False).head(20)
groupByTotalCost1.CustomerID = "CUSID " + groupByTotalCost1.CustomerID
groupByTotalCost1.head()
cf.set_config_file(theme='white')
groupByTotalCost1.iplot(kind = 'barh', x = 'CustomerID', y = 'PurchaseTotal', title = 'Top 20 Customer ID by Purchases',
                     yTitle='Customer ID', xTitle='Purchase Total')
sns.set(style="whitegrid")
ax1 = sns.boxplot(x=data["Quantity"],orient="h", palette="Pastel2")
#Outliers InterQuartile Range (IQR) 1.5 IQR Rule for Numeric Data
def outlierDetection(datacolumn):
    #Sort the data in ascending order
    #GET Q1 and Q3
    sorted(datacolumn)
    Q1,Q3 = np.percentile(datacolumn, [25,75])
    
    #Calc IQR
    IQR = Q3 - Q1
    
    #Calc LowerRange
    lr = Q1 - (1.5 * IQR)
    #Calc Upper Range
    ur = Q3 + (1.5 * IQR)
    #return 1,2
    return lr,ur

QuantityOutliersDataFrame = pd.DataFrame(columns=['FeatureUniqueValues',
                                         'lowerRange','upperRange','OutlierLower','OutlierUpper','OutlierFoundStatus'])
lowerRange,upperRange = outlierDetection(data['Quantity'])
outlier_upper = data['Quantity'] > upperRange 
outlier_lower = data['Quantity'] < lowerRange
if outlier_upper.any() or outlier_lower.any():
    OutlierFoundStatus = True
else:
    OutlierFoundStatus = False
QuantityOutliersDataFrame = QuantityOutliersDataFrame.append({'FeatureUniqueValues': data['Quantity'].nunique(),
                                                              'lowerRange' : lowerRange, 
                                                              'upperRange' : upperRange,
                                                              'OutlierLower' : data['Quantity'].min(),
                                                              'OutlierUpper' :data['Quantity'].max(),
                                                              'OutlierFoundStatus' : OutlierFoundStatus
                                                             }, ignore_index=True)
QuantityOutliersDataFrame
dataQuantity = pd.DataFrame(columns=['Type','Count'])
dataQuantity = dataQuantity.append({'Type': "Negative Values",'Count' : (data.Quantity < 0).sum()}, 
                                       ignore_index=True)
dataQuantity = dataQuantity.append({'Type': "Zero Values",'Count' : (data.Quantity == 0).sum()}, 
                                       ignore_index=True)
dataQuantity = dataQuantity.append({'Type': "Positive Values",'Count' : (data.Quantity > 0).sum()}, 
                                       ignore_index=True)

# Set notebook mode to work in offline
pyo.init_notebook_mode()
# Use `hole` to create a donut-like pie chart
fig = go.Figure(data=[go.Pie(labels=dataQuantity.Type, values=dataQuantity.Count, title = "Quantity Classification",hole=.8)])
fig.show()
data[(data.Quantity < -20000) | (data.Quantity > 20000)]
# We don't need records with Quantity < 0 and positive outliers which are cancelled for RFM analysis 
#so let's remove those records
data5 = data[(data.Quantity <=0) | (data.Quantity > 20000)]
data5.reset_index(drop=True,inplace = True)

data = data[(data.Quantity>0) & (data.Quantity <= 20000)]
data.reset_index(drop=True,inplace = True)
data.shape
sns.set(style="whitegrid")
ax1 = sns.boxplot(x=data["Quantity"],orient="h", palette="Pastel2")
sns.set(style="whitegrid")
ax1 = sns.boxplot(x=data["UnitPrice"],orient="h", palette="Pastel2")
UnitPriceOutliersDataFrame = pd.DataFrame(columns=['FeatureUniqueValues',
                                         'lowerRange','upperRange','OutlierLower','OutlierUpper','OutlierFoundStatus'])
lowerRange,upperRange = outlierDetection(data['UnitPrice'])
outlier_upper = data['UnitPrice'] > upperRange 
outlier_lower = data['UnitPrice'] < lowerRange
if outlier_upper.any() or outlier_lower.any():
    OutlierFoundStatus = True
else:
    OutlierFoundStatus = False
UnitPriceOutliersDataFrame = UnitPriceOutliersDataFrame.append({'FeatureUniqueValues': data['UnitPrice'].nunique(),
                                                              'lowerRange' : lowerRange, 
                                                              'upperRange' : upperRange,
                                                              'OutlierLower' : data['UnitPrice'].min(),
                                                              'OutlierUpper' :data['UnitPrice'].max(),
                                                              'OutlierFoundStatus' : OutlierFoundStatus
                                                             }, ignore_index=True)
UnitPriceOutliersDataFrame
dataUnitPrice = pd.DataFrame(columns=['Type','Count'])
dataUnitPrice = dataUnitPrice.append({'Type': "Negative Values",'Count' : (data.UnitPrice < 0).sum()}, 
                                       ignore_index=True)
dataUnitPrice = dataUnitPrice.append({'Type': "Zero Values",'Count' : (data.UnitPrice == 0).sum()}, 
                                       ignore_index=True)
dataUnitPrice = dataUnitPrice.append({'Type': "Positive Values",'Count' : (data.UnitPrice > 0).sum()}, 
                                       ignore_index=True)

# Set notebook mode to work in offline
pyo.init_notebook_mode()
# Use `hole` to create a donut-like pie chart
fig = go.Figure(data=[go.Pie(labels=dataUnitPrice.Type, values=dataUnitPrice.Count, 
                             title = "UnitPrice Classification",hole=.8)])
fig.show()
# We don't need records with UnitPrice < 0 for RFM analysis so let's remove those records
data = data[data.UnitPrice > 0]
data.reset_index(drop=True,inplace = True)
data.shape
sns.set(style="whitegrid")
ax1 = sns.boxplot(x=data["UnitPrice"],orient="h", palette="Pastel2")
dataInvoiceDate = pd.DataFrame(columns=['Type','Count'])
dataInvoiceDate = dataInvoiceDate.append({'Type': "From December 2011",'Count' 
                                          : (data.InvoiceDate >= "2011-12-01 00:00:00").sum()}, ignore_index=True)
dataInvoiceDate = dataInvoiceDate.append({'Type': "Before December 2011",'Count' 
                                          : (data.InvoiceDate < "2011-12-01 00:00:00").sum()},ignore_index=True)
# Set notebook mode to work in offline
pyo.init_notebook_mode()
# Use `hole` to create a donut-like pie chart
fig = go.Figure(data=[go.Pie(labels=dataInvoiceDate.Type, values=dataInvoiceDate.Count, 
                             title = "Invoice Date Classification",hole=.8)])
fig.show()
#We need only 1 year of data, removing data of December 2011 which has partial month data
data = data[data.InvoiceDate < "2011-12-01 00:00:00"]
data.reset_index(drop=True,inplace = True)
data.shape
groupByTotalCost3 = data.groupby(["CustomerID"], as_index=False)['TotalCost'].sum().rename(columns={'CustomerID':'CustomerID','TotalCost' : 'PurchaseTotal'})
groupByTotalCost3.CustomerID = groupByTotalCost3.CustomerID.astype(str)
groupByTotalCost3 = groupByTotalCost3.sort_values(by='PurchaseTotal', ascending=False).head(20)
groupByTotalCost3.CustomerID = "CUSID " + groupByTotalCost3.CustomerID
groupByTotalCost3.head()
cf.set_config_file(theme='white')
groupByTotalCost3.iplot(kind = 'barh', x = 'CustomerID', y = 'PurchaseTotal', title = 'Top 20 Customer ID by Purchases After EDA',
                     yTitle='Customer ID', xTitle='Purchase Total')
countries.info()
countries[countries.Country_Code.isna()]
countries = countries.dropna()
countries.reset_index(drop=True,inplace = True)
countries.shape
#Create Dataframe dataTableau
dataTableau = pd.merge(data,countries,on="Country", how='left')
dataTableau.head()                
dataTableau['Region'] = 'Others'
dataTableau.Region[dataTableau.Country =="United Kingdom"] = 'UK'
dataTableau.head()
##### Create MonthYear Column for Visualization
def obtain_TableauMonthYear(InvoiceDate):
    return dt.datetime(InvoiceDate.year,InvoiceDate.month,1) 

##### Compute column InvoiceMonth column applying function obtain_CohortMonth on the datetime column InvoiceDate
dataTableau['InvoiceMonth'] = dataTableau['InvoiceDate'].apply(obtain_TableauMonthYear) 
dataTableau['MonthYear'] = dataTableau['InvoiceMonth'].dt.strftime('%Y-%m')
dataTableau.drop(['InvoiceMonth'], axis=1, inplace=True)
dataTableau.head()
dataTableau.info()
dataRefundCalc = data5[(data5.Quantity <=0)]
dataRefundCalc.reset_index(drop=True,inplace = True)
groupByNegativeInvoices = dataRefundCalc.groupby(["InvoiceNo"], as_index=False)['TotalCost'].sum().rename(columns={'InvoiceNo':'InvoiceNo','TotalCost' : 'RefundValue'})
groupByNegativeInvoices = groupByNegativeInvoices.sort_values(by='RefundValue', ascending=False).head(20)
groupByNegativeInvoices.shape
refundedInvoices = pd.DataFrame(columns=['RefundedInvoices'])
refundedInvoices = refundedInvoices.append({'RefundedInvoices': len(groupByNegativeInvoices)},ignore_index=True)
refundedInvoices
cohortdata = data.copy()
cohortdata.shape
cohortdata.head()
##### Date Parsing Function obtain_CohortMonth
def obtain_CohortMonth(InvoiceDate):
    return dt.datetime(InvoiceDate.year,InvoiceDate.month,1) 

##### Compute column InvoiceMonth column applying function obtain_CohortMonth on the datetime column InvoiceDate
cohortdata['InvoiceMonth'] = cohortdata['InvoiceDate'].apply(obtain_CohortMonth) 

###### Create a Grouping cohortGrouping on the CustomerID and InvoiceMonth
cohortGrouping = cohortdata.groupby('CustomerID')['InvoiceMonth'] 

###### Compute the CohortMonth (month of first transaction) for each Customer by identifying first transaction
cohortdata['CohortMonth'] = cohortGrouping.transform('min')

cohortdata.head()
def obtainDateInterval(dataframe, column):
    year = dataframe[column].dt.year
    month = dataframe[column].dt.month
    return year, month
# Parse Year & Month of InvoiceMonth column into invoiceYear & invoiceMonth columns using function obtainDateInterval
invoiceYear, invoiceMonth = obtainDateInterval(cohortdata,'InvoiceMonth')

# Parse Year & Month of CohortMonth column into cohortYear & cohortMonth columns using function obtainDateInterval
cohortYear, cohortMonth = obtainDateInterval(cohortdata,'CohortMonth')
# Calculate difference between invoiceYear and the assigned cohortYear in years for every row item
diffYears = invoiceYear - cohortYear

# Calculate difference between invoiceMonth and the assigned cohortMonth in months for every row item
diffMonths = invoiceMonth - cohortMonth

# Set CohortIndex as diffYears*12 + diffMonths + 1 : 
#1 is added so that CohortIndex is never 0 if the first and subsequent transaction for customer are in same month
cohortdata['CohortIndex'] = diffYears.mul(12) + diffMonths + 1

cohortdata.head()
#Create Individual Months List for cohort analysis
mapMonths = cohortdata.groupby(["InvoiceMonth"], as_index=False)['TotalCost'].sum().rename(columns={'InvoiceMonth':'InvoiceMonth','TotalCost' : 'PurchaseTotal'})
mapMonths =mapMonths.sort_values(by='InvoiceMonth')
mapMonths.reset_index(drop=True,inplace = True)
mapMonths['MonthYear'] = mapMonths['InvoiceMonth'].dt.strftime('%b-%Y')
mapMonthsList = mapMonths['MonthYear'].to_list()
mapMonthsList
# Create a groupby object MonthlyActiveCustomerGroup and pass the monthly cohort and cohort index as a list
MonthlyActiveCustomerGroup = cohortdata.groupby(['CohortMonth', 'CohortIndex']) 

# Calculate the sum of the TotalCost column
MonthlyCustomers = MonthlyActiveCustomerGroup['CustomerID'].apply(pd.Series.nunique)

# Reset the index of cohort_data
MonthlyCustomers = MonthlyCustomers.reset_index()

# Create a pivot 
TotalMonthlyCustomers = MonthlyCustomers.pivot(index='CohortMonth', columns='CohortIndex', values='CustomerID')

# Display Monthly Customer Cohort count
TotalMonthlyCustomers
# Initialize plot figure
plt.figure(figsize=(20, 15))

# Add a title
plt.title('Active Customer Count by Cohort Month',fontsize=18)

# Create the heatmap
sns.heatmap(data = TotalMonthlyCustomers,
            annot=True,
            vmin = 0.0,
            cmap='Pastel2',
            vmax = list(TotalMonthlyCustomers.max().sort_values(ascending = False))[1]+3,
            fmt = '.1f',
            linewidth = 0.3,
            yticklabels=mapMonthsList)
plt.show();
# Create a groupby object TotalCostGroup and pass the monthly cohort and cohort index as a list
TotalCostGroup = cohortdata.groupby(['CohortMonth', 'CohortIndex']) 

# Calculate the sum of the TotalCost column
PurchaseTotal = TotalCostGroup['TotalCost'].sum()

# Reset the index of cohort_data
PurchaseTotal = PurchaseTotal.reset_index()

# Create a pivot 
TotalPurchase = PurchaseTotal.pivot(index='CohortMonth', columns='CohortIndex', values='TotalCost')
TotalPurchase
cf.set_config_file(theme='white')
mapMonths.iplot(kind = 'barh', x = 'MonthYear', y = 'PurchaseTotal', title = 'Purchases By Month',
                     yTitle='Monthly Purchases in £', xTitle='Month Year of Purchase',text = 'PurchaseTotal')
# Initialize plot figure
plt.figure(figsize=(20, 15))

# Add a title
plt.title('Total Items Purchased Spread by Cohort Month',fontsize=18)

# Create the heatmap
sns.heatmap(data = TotalPurchase,
            annot=True,
            vmin = 0.0,
            cmap='Pastel2',
            vmax = list(TotalPurchase.max().sort_values(ascending = False))[1]+3,
            fmt = '.1f',
            linewidth = 0.3,
            yticklabels=mapMonthsList)
plt.show();
# Create a groupby quantityGroup object and pass the monthly cohort and cohort index as a list
quantityGroup = cohortdata.groupby(['CohortMonth', 'CohortIndex']) 

# Calculate the average of the Quantity column
quantityMean = quantityGroup['Quantity'].mean()

# Reset the index of cohort_data
quantityMean = quantityMean.reset_index()

# Create a pivot 
MeanQuantity = quantityMean.pivot(index='CohortMonth', columns='CohortIndex', values='Quantity')

# Initialize plot figure
plt.figure(figsize=(15, 12))

# Add a title
plt.title('Mean Quantity of Items Purchased Spread by Cohort Month',fontsize=18)

# Create the heatmap
sns.heatmap(data = MeanQuantity,
            annot=True,
            vmin = 0.0,
            cmap='Pastel2',
            vmax = list(MeanQuantity.max().sort_values(ascending = False))[1]+3,
            fmt = '.1f',
            linewidth = 0.3,
            yticklabels=mapMonthsList)
plt.show();
cohortGroup = cohortdata.groupby(['CohortMonth', 'CohortIndex'])
# Count the number of unique values per customer ID
cohort_data = cohortGroup['CustomerID'].apply(pd.Series.nunique).reset_index()

# Create a pivot 
cohortCounts = cohort_data.pivot(index='CohortMonth', columns='CohortIndex', values='CustomerID')

# Select the first column and store it to cohort_sizes
cohortSizes = cohortCounts.iloc[:,0]

# Divide the cohort count by cohort sizes along the rows
retention = cohortCounts.divide(cohortSizes, axis=0).mul(100)
# Initialize inches plot figure
plt.figure(figsize=(15,15))

# Add a title
plt.title('Customer Retention Rate Spread by Cohort Month in Percentage (%)',fontsize=18)

# Create the heatmap
sns.heatmap(data=retention,
            annot = True,
            cmap = "Pastel2",
            vmin = 0.0,
            vmax = list(retention.max().sort_values(ascending = False))[1]+3,
            fmt = '.1f',
            linewidth = 0.3,
            yticklabels=mapMonthsList)
plt.show();
RFMData = data.copy()
RFMData.shape
RFMData.head()
# Set variable current_date to this max of Invoice date in dataframe RFMData
currentDateTime = RFMData['InvoiceDate'].max() +timedelta(days=1)
currentYear = currentDateTime.year
currentMonth = currentDateTime.month
currentDay = currentDateTime.day
currentDate = dt.date(currentYear,currentMonth,currentDay)
currentDate
# Lets create a date column RecentPurchaseDate for date part only of InvoiceDate
RFMData['RecentPurchaseDate'] = RFMData.InvoiceDate.dt.date
RFMData.head()
recency = RFMData.groupby('CustomerID')['RecentPurchaseDate'].max().reset_index()
recency.head()
# Create column currentDate in dataframe recency
recency['CurrentDate'] = currentDate
recency.head()
# Compute Recency as difference between current date and RecentPurchaseDate
recency['Recency'] = recency.RecentPurchaseDate.apply(lambda elapsed: (currentDate - elapsed).days)
recency.head()
# Data Clean Up - Drop Columns RecentPurchaseDate and CurrentDate
recency.drop(['RecentPurchaseDate','CurrentDate'], axis=1, inplace=True)
recency.head()
recency.shape
frequency = RFMData.groupby('CustomerID').InvoiceNo.count().reset_index().rename(columns={'InvoiceNo':'Frequency'})
frequency.head()
frequency.shape
monetary = RFMData.groupby('CustomerID').TotalCost.sum().reset_index().rename(columns={'TotalCost':'Monetary'})
monetary.head()
monetary.shape
intermediateRFMerge = recency.merge(frequency, on='CustomerID')
RFMModel = intermediateRFMerge.merge(monetary, on='CustomerID')
RFMModel.head()
RFMModel.shape
data.CustomerID.unique().shape
RFMModel.set_index('CustomerID',inplace=True)
RFMModel.head()
# Match RFMData Customer ID and RFMModel Index and display the head of the comparison result in RFMData
RFMData[RFMData.CustomerID == RFMModel.index[0]].head()
# Check if the number difference of days from the purchase date in original record is same as shown in RFMModel
(currentDate - RFMData[RFMData.CustomerID == RFMModel.index[0]].iloc[0].RecentPurchaseDate).days == RFMModel.iloc[0,0]
# RFM Quartiles
RFMQuantiles = RFMModel.quantile(q=[0.25,0.5,0.75]).to_dict()
RFMQuantiles
RFMQuantilesDF = pd.DataFrame(RFMModel.quantile(q=[0,0.25,0.5,0.75,1]))
RFMQuantilesDF
plt.figure(figsize=(15, 10))
sns.set(style="whitegrid")
ax1 = sns.boxplot(x=RFMQuantilesDF["Recency"],orient="h", palette="Pastel2")
# Set title
plt.title('Recency Quantiles Distribution Boxplot',fontsize=18)
plt.figure(figsize=(15, 10))
sns.set(style="whitegrid")
ax1 = sns.boxplot(x=RFMQuantilesDF["Frequency"],orient="h", palette="Pastel2")
# Set title
plt.title('Frequency Quantiles Distribution Boxplot',fontsize=18)
plt.figure(figsize=(15, 10))
sns.set(style="whitegrid")
ax1 = sns.boxplot(x=RFMQuantilesDF["Monetary"],orient="h", palette="Pastel2")
# Set title
plt.title('Monetary Quantiles Distribution Boxplot',fontsize=18)
RFMQuantilesDFSubPlot = RFMQuantilesDF.head(4).plot.bar(rot=0, subplots=True,figsize=(12, 8))
RFMQuantilesDFSubPlot[1].legend(loc=2)  
# Set title
plt.title('Quantiles Distribution Bar Spread 0 - 25 - 50 - 75',fontsize=18)
RFMSegment = RFMModel.copy()
RFMSegment.shape
RFMSegment.head()
# RScore Function Arguments Getting Passed (S = Score, P = Recency, Q = quantiles dict)
def RScore(S,P,Q):
    if S > Q[P][0.75]:
        return 1
    elif S > Q[P][0.50]:
        return 2
    elif S > Q[P][0.25]: 
        return 3
    else:
        return 4
    
RFMSegment['R'] = RFMSegment['Recency'].apply(RScore, args=('Recency',RFMQuantiles,))
RFMSegment.head()
# FMScore Function Arguments Getting Passed (S = Score, P = Frequency/Monetary, Q = quantiles dict)
def FMScore(S,P,Q):
    if S > Q[P][0.75]:
        return 4
    elif S > Q[P][0.50]:
        return 3
    elif S > Q[P][0.25]: 
        return 2
    else:
        return 1
    
RFMSegment['F'] = RFMSegment['Frequency'].apply(FMScore, args=('Frequency',RFMQuantiles,))
RFMSegment.head()
RFMSegment['M'] = RFMSegment['Monetary'].apply(FMScore, args=('Monetary',RFMQuantiles,))
RFMSegment.head()
# Compute RFM Score as a String
RFMSegment['RFM_Segment'] = RFMSegment.R.astype(str) +  RFMSegment.F.astype(str) + RFMSegment.M.astype(str)
RFMSegment['RFM_Score'] = RFMSegment[['R', 'F', 'M']].sum(axis = 1)
RFMSegment.head()
print(RFMSegment['RFM_Segment'].unique())
RFMSegment.info()
# Reset the index to create customerID column
RFMSegment.reset_index(inplace=True)
RFMSegment.shape
RFMSegment.head()
#Create Customer Segment Function
def segment_customer(Dataframe):
    if Dataframe['R'] >3:
        if Dataframe['F'] >3:
            if Dataframe['M'] >3:
                return 'Best Customer'
            elif Dataframe['M'] >2:
                return 'Medium/High Spending Active Loyal Customer'
            else:
                return 'Low-Spending Active Loyal Customer'
        elif Dataframe['F'] >2:
            if Dataframe['M'] >2:
                return 'Medium/High Spending Active Loyal Customer'
            else:
                return 'Low-Spending Active Loyal Customer'
        else:
            if Dataframe['M'] >2:
                return 'Medium/High Spending Active Customer'
            else:
                return 'Low-Spending New Customer'
    elif Dataframe['R'] >2:
        if Dataframe['F'] >3:
            if Dataframe['M'] >3:
                return 'Inactive Best Customer'
            elif Dataframe['M'] >2:
                return 'Medium/High Spending Active Loyal Customer'
            else:
                return 'Low-Spending Inactive Loyal Customer'
        elif Dataframe['F'] >2:
            if Dataframe['M'] >2:
                return 'Medium/High Spending Active Loyal Customer'
            else:
                return 'Low-Spending Inactive Loyal Customer'
        else:
            if Dataframe['M'] >2:
                return 'Medium/High Spending Active Customer'
            else:
                return 'Low Spending Inactive Customer'
    else:
        if Dataframe['F'] >3:
            if Dataframe['M'] >3:
                return 'Churned Best Customer'
            elif Dataframe['M'] >2:
                return 'Churned Loyal Customer'
            else:
                return 'Churned Frequent Customer'
        elif Dataframe['F'] >2:
            if Dataframe['M'] >2:
                return 'Churned Loyal Customer'
            else:
                return 'Churned Frequent Customer'
        else:
            if Dataframe['M'] >2:
                return 'Churned Medium-High Spender'
            else:
                return 'Churned Low Spending Infrequent Customer'   
RFMSegment['CustomerSegment'] = RFMSegment.apply(segment_customer, axis=1)
RFMSegment['CustomerSegment'].value_counts()
# Check head of RFMSegment Dataframe for Data Integrity
RFMSegment.head()
RFMSegment.CustomerSegment.value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
RFMSegment.sample(10)
#Create Market Segment Function
def segment_market(Dataframe):
    if Dataframe['RFM_Score'] > 11:
        return 'Platinum'
    elif Dataframe['RFM_Score'] > 8:
        return 'Gold'
    elif (Dataframe['RFM_Score'] > 5):
        return 'Silver'
    else:
        return 'Bronze'
RFMSegment['MarketSegment'] = RFMSegment.apply(segment_market, axis=1)
RFMSegment.groupby('MarketSegment').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': ['mean', 'count']}).round(1)
groupByCustomerSegment = RFMSegment.groupby(["CustomerSegment"], as_index=False)['CustomerID'].count().rename(columns={'CustomerSegment' : 'CustomerSegment','CustomerID':'NoofCustomers'})
groupByCustomerSegment.NoofCustomers = groupByCustomerSegment.NoofCustomers.astype(str)
groupByCustomerSegment = groupByCustomerSegment.sort_values(by='NoofCustomers', ascending=False)
groupByCustomerSegment.head()
cf.set_config_file(theme='white')
groupByCustomerSegment.iplot(kind = 'barh', x = 'CustomerSegment', y = 'NoofCustomers', title = 'Customer Segment Spread Categories',
                     yTitle='Customer Segment', xTitle='# of Customers',text = 'NoofCustomers',textposition = 'auto')
# Validate Distribution Skewness using distplot - subplots
fig, axes = plt.subplots(3, 1, figsize=(15, 15))
sns.distplot(RFMModel.Recency , color="chocolate", ax=axes[0], axlabel='Recency')
sns.distplot(RFMModel.Frequency, color="blue", ax=axes[1], axlabel='Frequency')
sns.distplot(RFMModel.Monetary, color="green", ax=axes[2], axlabel='Monetary')
# Set title
plt.title('Subplot Distribution to validate Skewness',fontsize=18)
plt.show();
# Let's describe the table to see if there are any negative values
RFMModel.describe()
# Transform the data before K-Means clustering


# Taking log first because normalization forces data for negative values
RFMModelLog = np.log(RFMModel)

# Normalize the data for uniform averages and means in the distribution.
scaler = StandardScaler()
RFMModelScaled = scaler.fit_transform(RFMModelLog)
RFMModelScaled = pd.DataFrame(data=RFMModelScaled, index=RFMModel.index, columns=RFMModel.columns)
RFMModelScaled.head()
# Revalidate Distribution Subplot to check data skewness after log transformation
fig, axes = plt.subplots(3, 1, figsize=(15, 15))
sns.distplot(RFMModelScaled.Recency , color="chocolate", ax=axes[0], axlabel='Recency')
sns.distplot(RFMModelScaled.Frequency , color="blue", ax=axes[1], axlabel='Frequency')
sns.distplot(RFMModelScaled.Monetary , color="green", ax=axes[2], axlabel='Monetary')
# Set title
plt.title('Subplot Distribution after Log Transformation',fontsize=18)
plt.show();
# Compute Within Cluster Sum of Squares(WCSS) Error through Elbow Method
WCSSKmeansOut = pd.DataFrame(columns=['Clusters','WCSSErrorScore'])
# Choose range of 1 to 14 as we have 13 Customer Segments
for loopCounter in range(1,14):
    kmeans = KMeans(n_clusters=loopCounter,random_state=1, init='k-means++')
    kmeans.fit(RFMModelScaled)
    WCSSKmeansOut = WCSSKmeansOut.append({'Clusters': loopCounter,'WCSSErrorScore' : kmeans.inertia_},ignore_index=True)
WCSSKmeansOut.set_index('Clusters',inplace=True)
WCSSKmeansOut
# Plot WCSS Elbow Graph
plt.figure(figsize=(20,15));
plt.title('WCSS Error Score Across Clusters 1 to 13 - Kmeans',fontsize=18)
WCSSKmeansOut.WCSSErrorScore.plot(marker='o')
#Optimum Cluster # Validation using Silhouette Score
#Using 3 as lower range number as typically best score for Silhouette is obtained for 2 clusters 
#With 13 segments selected - 2 clusters is not ideal
WCSSKmeansSilhouetteOut = pd.DataFrame(columns=['Clusters','silhouetteScore'])
for loopCounter2 in range(3,14):
    SilhouetteKMeansModel = KMeans(n_clusters=loopCounter2, random_state=1,init='k-means++').fit(RFMModelScaled)
    preds = SilhouetteKMeansModel.predict(RFMModelScaled)
    silhouetteScore = silhouette_score(RFMModelScaled,preds)
    WCSSKmeansSilhouetteOut = WCSSKmeansSilhouetteOut.append({'Clusters': loopCounter2,'silhouetteScore' : silhouetteScore},ignore_index=True)
# plot Silhouette graph
ax4 = sns.scatterplot(x="Clusters", y="silhouetteScore", data=WCSSKmeansSilhouetteOut)
# Set title
plt.title('Silhouette Score Across Clusters',fontsize=18)
WCSSKmeansSilhouetteOut
ComputedClusters = 4
#Build KMeans Model with 4 Clusters
KMeansModel = KMeans(n_clusters=ComputedClusters, random_state=1, init='k-means++')
KMeansModel.fit(RFMModelScaled)
clusterLabels = KMeansModel.labels_
clusterLabels
KMeansModel
clusterLabels.shape
RFMSegment.shape
# Assign the clusters as column to each customer
ClusterData = RFMSegment.assign(Cluster = clusterLabels)
ClusterData.head()
ClusterData.shape
# Compute counts of CustomerIDs assigned to different clusters
ClusterDataCount = ClusterData.Cluster.value_counts().sort_index().rename_axis('Cluster').reset_index(name='CustomerIDCount')
ClusterDataCount.Cluster = ClusterDataCount.astype(str)
ClusterDataCount.Cluster = "Cluster " + ClusterDataCount.Cluster
ClusterDataCount.head()
cf.set_config_file(theme='white')
ClusterDataCount.iplot(kind = 'barh', x = 'Cluster', y = 'CustomerIDCount', title = 'CustomerID Spread by Clusters',
                     xTitle='Cluster #', yTitle='# Of Customers')
#Sample the Cluster Table data to check data correctness
ClusterData.sample(10)
ClusterData[ClusterData.Cluster == 3].sample(10)
ClusterData[ClusterData.Cluster == 2].sample(10)
ClusterData[ClusterData.Cluster == 1].sample(10)
ClusterData[ClusterData.Cluster == 0].sample(10)
# Plot 2D plots of RF, FM and RM
TwoDPlot = RFMModelScaled.iloc[:,0:3].values
TwoDCount=TwoDPlot.shape[1]
for loopCounter01 in range(0,TwoDCount):
    for loopCounter02 in range(loopCounter01+1,TwoDCount):
        plt.figure(figsize=(20,10));
        plt.suptitle('Scatter Plot Visualization',fontsize=18)
        plt.scatter(TwoDPlot[clusterLabels == 0, loopCounter01], TwoDPlot[clusterLabels == 0, loopCounter02], s = 10, c = 'grey', label = 'Cluster0')
        plt.scatter(TwoDPlot[clusterLabels == 1, loopCounter01], TwoDPlot[clusterLabels == 1, loopCounter02], s = 10, c = 'chocolate', label = 'Cluster1')
        plt.scatter(TwoDPlot[clusterLabels == 2, loopCounter01], TwoDPlot[clusterLabels == 2, loopCounter02], s = 10, c = 'cyan', label = 'Cluster2')
        plt.scatter(TwoDPlot[clusterLabels == 3, loopCounter01], TwoDPlot[clusterLabels == 3, loopCounter02], s = 10, c = 'lightgreen', label = 'Cluster3')
        
        plt.scatter(KMeansModel.cluster_centers_[:,loopCounter01], KMeansModel.cluster_centers_[:,loopCounter02], s = 50, c = 'black', label = 'Centroids')
        plt.xlabel(RFMModelScaled.columns[loopCounter01])
        plt.ylabel(RFMModelScaled.columns[loopCounter02])
        plt.legend()       
        plt.show();
# Assign Cluster values to each customer in normalized dataframe
RFMModelScaled = RFMModelScaled.assign(Cluster = clusterLabels)

# Melt normalized dataframe into long form to have all metric in same column
RFMModelScaledMelt = pd.melt(RFMModelScaled.reset_index(),
                      id_vars=['CustomerID','Cluster'],
                      value_vars=['Recency', 'Frequency', 'Monetary'],
                      var_name='Metric',
                      value_name='Value')
RFMModelScaledMelt.head()
RFMModelScaledMelt.shape
# RFM Snake Plot Visualization
plt.figure(figsize=(15,10))
palette = sns.color_palette("Pastel2", 6)
sns.lineplot(x = 'Metric',
             y = 'Value',
             hue = 'Cluster',
             data = RFMModelScaledMelt,
             palette = "Pastel2")
plt.title("Snake Plot of RFM",fontsize=18)
plt.legend()
plt.show();
# Assign Cluster labels to RFMModelCluster table
RFMModelCluster = RFMModel.assign(Cluster = clusterLabels)

# Average attributes for each cluster
clusterMean = RFMModelCluster.groupby(['Cluster']).mean() 

# Calculate the population average
populationMean = RFMModel.mean()

# Calculate relative importance of attributes by 
AttributeInterDependence = (clusterMean / populationMean) - 1
AttributeInterDependence
plt.figure(figsize=(12, 5))
plt.title('Inter Dependence of RFM Attributes Across Clusters')
sns.heatmap(data=AttributeInterDependence, annot=True, fmt='.2f', cmap='Pastel2')
plt.show();
ClusterData[ClusterData.Cluster == 0].CustomerSegment.value_counts()
ClusterData[ClusterData.Cluster == 1].CustomerSegment.value_counts()
ClusterData[ClusterData.Cluster == 2].CustomerSegment.value_counts()
ClusterData[ClusterData.Cluster == 3].CustomerSegment.value_counts()
# Export datasets
#with pd.ExcelWriter(outputPath+'Online_Retail_EDA.xlsx',engine='xlsxwriter', mode='w') as writer:
with pd.ExcelWriter('Online_Retail_EDA.xlsx',engine='xlsxwriter', mode='w') as writer:
    dataTableau.to_excel(writer,sheet_name='Online_Retail',index=False)
    ClusterData.to_excel(writer,sheet_name='Cluster_Data',index=False)
    WCSSKmeansOut.to_excel(writer,sheet_name='WCSS_Data',index=False)
    WCSSKmeansSilhouetteOut.to_excel(writer,sheet_name='Silhouette_Data',index=False)
    refundedInvoices.to_excel(writer,sheet_name='Refunded_Invoices',index=False)