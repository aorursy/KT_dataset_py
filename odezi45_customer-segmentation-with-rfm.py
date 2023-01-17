import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from datetime import *
import matplotlib
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
%matplotlib inline
sales = pd.read_csv('/kaggle/input/online-retail-ii-uci/online_retail_II.csv')
#quick check on data column type
pd.DataFrame(sales.dtypes, columns=['Type'])
# convert InvoiceDate column to the right format
sales['InvoiceDate'] = pd.to_datetime(sales['InvoiceDate'])
# Let's check the descriptives without 'Model'
sales.describe(include='all')
# From the descriptive statistics, we can see some negative values in Quantity and Price, so Lets have a look
negprice = sales[sales['Price'] < 0]
negquantity = sales[sales['Quantity'] < 0]
print(negprice)
print(negquantity)
## take out negative price from the sales data
sales = sales[sales['Price']>= 0]

## we have also identify some descriptions that doesnt look like sales
sales2 = sales[sales['Description'].isin(['?',
'?????',
'back charges',
'bad quality',
'Came as green?',
'Came as green?',
'cant find',
'cant find',
'check',
'checked',
'checked',
'code mix up 72597',
'code mix up 72597',
'coding mix up',
'crushed',
'crushed',
'damaged',
'damaged/dirty',
'damaged?',
'damages',
'damages etc',
'damages, lost bits etc',
'damages?',
'damges',
'Damp and rusty',
'dirty',
'dirty, torn, thrown away.',
'display',
'entry error',
'faulty',
'for show',
'given away',
'gone',
'Gone',
'incorrect credit',
'lost',
'lost in space',
'lost?',
'missing',
'Missing',
'missing (wrongly coded?)',
'missing?',
'missings',
'reverse mistake',
'Rusty ',
'Rusty connections',
'show',
'show display',
'smashed',
'sold in wrong qnty',
'This is a test product.',
'used for show display',
'wet',
'wet & rotting',
'wet and rotting',
'wet cartons',
'wet ctn',
'wet damages',
'Wet, rusty-thrown away',
'wet/smashed/unsellable',
'wrong code',
'wrong ctn size',
'Zebra invcing error'])]
## so lets take those spurious sales out
sales = sales[~sales.apply(tuple,1).isin(sales2.apply(tuple,1))]
#Lets check for missing values
sales.isnull().sum()
## About 20% of the dataset has missing customer ID and 0.4% of the dataset has no description
# SO I willa ssume that the missing customer id are 9999 and the description is 'Unlnown'

sales[['Customer ID']] =sales[['Customer ID']].fillna(99999)
sales[['Description']] =sales[['Description']].fillna('Unknown')
sales.isnull().sum()
# lets also take out all negative quantity as, they are either returns or errors in the data.
sales = sales[sales['Quantity'] > 0]

#sales['ordertype'] = np.where(sales['Quantity'] < 0,'sale','return')
# Let's check the descriptives without 'Model'
sales.describe(include='all')
## Now Lets find the first and second time a customer ordered by aggregating the values
sales_ = sales.groupby('Invoice').agg(
    Customer =('Customer ID', 'first'),
    InvoiceDate2=('InvoiceDate', 'min'))
sales_.reset_index(inplace = True)
sales_['daterank'] = sales_.groupby('Customer')['InvoiceDate2'].rank(method="first", ascending=True)

# find customers second purchase and name dataframe sales_
sales_ = sales_[sales_['daterank']== 2]
sales_.drop(['Invoice', 'daterank'], axis=1, inplace=True)
sales_.columns = ['Customer ID', 'InvoiceDate2']
sales_
# Lets Aggregate the data to find certain customer metrics 
sales['amount'] = sales['Price'] * sales['Quantity']
salesgroup = sales.groupby('Customer ID').agg(
    Country=('Country', 'first'),
    sum_price=('Price', 'sum'),
    sum_quantity=('Quantity', 'sum'),
    max_date=('InvoiceDate', 'max'),
    min_date=('InvoiceDate', 'min'),
    count_order=('Invoice', 'nunique'),
    avgitemprice=('Price', 'mean'),
    monetary =('amount', 'sum'),
    count_product=('Invoice', 'count'))

salesgroup.reset_index(inplace = True)
salesgroup
#Find the max date of this study
maxdate = sales['InvoiceDate'].max()

#Calculate AOV. Item per basket
salesgroup['avgordervalue'] = salesgroup['monetary']/salesgroup['count_order']
salesgroup['itemsperbasket'] = salesgroup['sum_quantity']/salesgroup['count_order']

# join the data with the dataframe containing customer id with 2nd visits
salesgroup = pd.merge(salesgroup, sales_ , how='left', on=['Customer ID'])
salesgroup
# find difference between first purchase and 2nd purchase 
salesgroup['daysreturn']  = salesgroup['InvoiceDate2']- salesgroup['min_date']
salesgroup['daysreturn'] = salesgroup['daysreturn']/np.timedelta64(1,'D')
salesgroup['daysmaxmin']  = salesgroup['max_date']- salesgroup['min_date']
salesgroup['daysmaxmin'] = (salesgroup['daysmaxmin']/np.timedelta64(1,'D')) + 1
salesgroup
#calculate Frequency and Recency
salesgroup['frequency'] = np.where(salesgroup['count_order'] >1,salesgroup['count_order']/salesgroup['daysmaxmin'],0)
salesgroup['recency']  = maxdate- salesgroup['max_date']
salesgroup['recency'] = salesgroup['recency']/np.timedelta64(1,'D')
salesgroup
salesgroup.describe(include='all')
# Now we have the values for Recency, Frequency and Monetary parameters. Each customer will get a note between 1 and 4 for each parameter.
#By Applying quantile method we group each quantile into 25% of the population. 

#so letsdefine the quantile and save it ina dictionary
quintiles = salesgroup[['recency', 'frequency', 'monetary']].quantile([.25, .50, .75]).to_dict()
quintiles2 = salesgroup[['recency', 'frequency', 'monetary']].quantile([.2, .4, 0.6, .8]).to_dict()
quintiles
# Create a fuction that assign ranks from 1 to 4. 
# A smaller Recency value is better
# For Frequency and Monetary values, a Higher value is better. 
# so we have two different functions.
def r_score(x):
    if x <= quintiles['recency'][.25]:
        return 4
    elif x <= quintiles['recency'][.50]:
        return 3
    elif x <= quintiles['recency'][.75]:
        return 2
    else:
        return 1
    
def fm_score(x, c):
    if x <= quintiles[c][.25]:
        return 1
    elif x <= quintiles[c][.50]:
        return 2
    elif x <= quintiles[c][.75]:
        return 3
    else:
        return 4    
#lets get the RFM values by calling the function above

salesgroup['R'] = salesgroup['recency'].apply(lambda x: r_score(x))
salesgroup['F'] = salesgroup['frequency'].apply(lambda x: fm_score(x, 'frequency'))
salesgroup['M'] = salesgroup['monetary'].apply(lambda x: fm_score(x, 'monetary'))

salesgroup['RFM Score'] = salesgroup['R'].map(str) + salesgroup['F'].map(str) + salesgroup['M'].map(str)
salesgroup['RFM Score'] = salesgroup['RFM Score'].astype(int)
pd.DataFrame(salesgroup.dtypes, columns=['Type'])
salesgroup['RFM Score'] = salesgroup['RFM Score'].astype(int)
def rrr(salesgroup):
    if salesgroup['RFM Score'] == 111 :
        d = 'Best Customers'
    elif salesgroup['RFM Score'] == 112 :
        d = 'High Spending New Customers'
    elif salesgroup['RFM Score'] == 113 :
        d = 'Lowest Spending Active Lyal Customers'
    elif salesgroup['RFM Score'] == 114 :
        d = 'Lowest Spending Active Lyal Customers'
    elif salesgroup['RFM Score'] == 422 :
        d = 'Churned Best Customers'
    elif salesgroup['RFM Score'] == 421 :
        d = 'Churned Best Customers'
    elif salesgroup['RFM Score'] == 412 :
        d = 'Churned Best Customers'
    elif salesgroup['RFM Score'] == 411 :
        d = 'Churned Best Customers'
    else:
        d = 'Unclassed'
    return d

salesgroup['comms_label'] = salesgroup.apply(rrr, axis=1)
salesgroup
def www(salesgroup):
    if salesgroup['RFM Score'] == 111 : 
        d = 'Core'
    elif salesgroup['F'] == 1 : 
        d = 'Loyal'
    elif salesgroup['M'] == 1 : 
        d = 'Whales'
    elif salesgroup['F'] == 1 &  salesgroup['M'] == 3: 
        d = 'Promising'
    elif salesgroup['F'] == 1 &  salesgroup['M'] == 4: 
        d = 'Promising'
    elif salesgroup['R'] == 1 & salesgroup['F'] == 4: 
        d = 'Rookies'
    elif salesgroup['R'] == 4 & salesgroup['F'] == 4 : 
        d = 'Slipping'
    else:
        d = 'Unclassed'
    return d

salesgroup['sales_label'] = salesgroup.apply(www, axis=1)
salesgroup
## Quick Plot of the count of customer ID in various classification based on the first instance which was Marketting

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 15
fig_size[1] = 8
plt.rcParams["figure.figsize"] = fig_size

sns.set(style="darkgrid")
ax = sns.countplot(x="sales_label", data=salesgroup)

## Quick Plot of the count of customer ID in various classification based on the second instance which is customer Insight
ax = sns.countplot(x="comms_label", data=salesgroup)
#so letsdefine the quantile and save it ina dictionary
quintiles2 = salesgroup[['recency', 'frequency', 'monetary']].quantile([.2, .4, 0.6, .8]).to_dict()

def r_score2(y):
    if y <= quintiles2['recency'][.2]:
        return 1
    elif y <= quintiles2['recency'][.4]:
        return 2
    elif y <= quintiles2['recency'][.6]:
        return 3
    elif y <= quintiles2['recency'][.8]:
        return 4
    else:
        return 5
    
def fm_score2(y, k):
    if y <= quintiles2[k][.2]:
        return 1
    elif y <= quintiles2[k][.4]:
        return 2
    elif y <= quintiles2[k][.6]:
        return 3
    elif y <= quintiles2[k][.8]:
        return 4
    else:
        return 5    
#lets get the RFM values by calling the function above

salesgroup['R2'] = salesgroup['recency'].apply(lambda y: r_score2(y))
salesgroup['F2'] = salesgroup['frequency'].apply(lambda y: fm_score2(y, 'frequency'))
salesgroup['M2'] = salesgroup['monetary'].apply(lambda y: fm_score2(y, 'monetary'))

salesgroup['RFM Score2'] = salesgroup['R2'].map(str) + salesgroup['F2'].map(str) + salesgroup['M2'].map(str)
salesgroup['RFM Score2'] = salesgroup['RFM Score2'].astype(int)

##So lets group the customersinto 11 based on RFM scores.

def mapl(salesgroup, r_rule, fm_rule, label, colname='new_label'):
    salesgroup.loc[(salesgroup['R2'].between(r_rule[0], r_rule[1]))
            & (salesgroup['F2'].between(fm_rule[0], fm_rule[1])), colname] = label
    return salesgroup

salesgroup['new_label'] = ''

salesgroup = mapl(salesgroup, (4,5), (4,5), 'Champions')
salesgroup = mapl(salesgroup, (2,5), (3,5), 'Loyal customers')
salesgroup = mapl(salesgroup, (3,5), (1,3), 'Potential loyalist')
salesgroup = mapl(salesgroup, (4,5), (0,1), 'New customers')
salesgroup = mapl(salesgroup, (3,4), (0,1), 'Promising')
salesgroup = mapl(salesgroup, (2,3), (2,3), 'Needing attention')
salesgroup = mapl(salesgroup, (2,3), (0,2), 'About to sleep')
salesgroup = mapl(salesgroup, (0,2), (2,5), 'At risk')
salesgroup = mapl(salesgroup, (0,1), (4,5), 'Cant loose them')
salesgroup = mapl(salesgroup, (1,2), (1,2), 'Hibernating')
salesgroup = mapl(salesgroup, (0,2), (0,2), 'Lost')


customercategory = salesgroup.groupby('new_label').agg(
    count=('Customer ID', 'count'))

customercategory.reset_index(inplace = True)
customercategory.columns.values
# lets visualise the new cluster
import squarify 
 
#Utilise matplotlib to scale our goal numbers between the min and max, then assign this scale to our values.
norm = matplotlib.colors.Normalize(vmin=min(customercategory['count']), vmax=max(customercategory['count']))
colors = [matplotlib.cm.Blues(norm(value)) for value in customercategory['count']]

#Create our plot and resize it.
fig = plt.gcf()
ax = fig.add_subplot()
fig.set_size_inches(12, 6)

#Use squarify to plot our data, label it and add colours. We add an alpha layer to ensure black labels show through
squarify.plot(label=customercategory['new_label'],sizes= customercategory['count'], color = colors, alpha=.6)
plt.title("Customer Category Groupings based on RFM ",fontsize=23,fontweight="bold")

#Remove our axes and display the plot
plt.axis('off')
plt.show()
sales_df = salesgroup.drop(columns=['max_date','min_date', 'R', 'F', 'M' ])
# Get the 10 RFM score with the most customers
Top10_RFM = salesgroup['RFM Score2'].value_counts()[:10].index.tolist()
Top10_RFM 
sakesviz = salesgroup[['Customer ID', 'Country', 'monetary','frequency','count_product']]
#top ten frequent buyer
sakesviz.nlargest(10,'frequency')

#top ten volume of Item buyer
sakesviz.nlargest(10, 'count_product')
#Top10_Customer_LTV
sakesviz.nlargest(10,'monetary')
# Summary metrics for the 10 most popular RFM
Top10_RFM_summary = salesgroup[salesgroup['RFM Score2'].isin(Top10_RFM)].groupby('RFM Score2').agg(
    mean_recency=('recency', 'mean'),
    std_recency=('recency', 'std'),
    mean_frequency=('frequency', 'mean'),
    std_frequency=('frequency', 'std'),
    mean_monetary=('monetary', 'mean'),
    std_monetary=('monetary', 'std'),   
    samples=('Customer ID', lambda x: len(x)*100/len(salesgroup['new_label']))
).round(2)

Top10_RFM_summary.reset_index(inplace = True)

Top10_RFM_summary
# Summary metrics per RFM Category
Category_summary = salesgroup.groupby('new_label').agg(
    mean_recency=('recency', 'mean'),
    std_recency=('recency', 'std'),
    mean_frequency=('frequency', 'mean'),
    std_frequency=('frequency', 'std'),
    mean_monetary=('monetary', 'mean'),
    std_monetary=('monetary', 'std'),  
    samples_percentage=('Customer ID', lambda x: len(x)*100/len(salesgroup['RFM Score2']))
).round(2)

Category_summary.reset_index(inplace = True)
Category_summary
print(plt.rcParams.get('figure.figsize'))


plt.xticks(range(len(Category_summary['mean_recency'])), Category_summary['new_label'])
plt.xlabel('Customer Categories')
plt.ylabel('Mean Recency')
plt.title('Mean Recency by Customer Categoriess')
plt.bar(range(len(Category_summary['mean_recency'])), Category_summary['mean_recency']) 
plt.show()


plt.xticks(range(len(Category_summary['mean_frequency'])), Category_summary['new_label'])
plt.xlabel('Customer Categories')
plt.ylabel('Mean frequency')
plt.title('Mean frequency by Customer Categoriess')
plt.bar(range(len(Category_summary['mean_frequency'])), Category_summary['mean_frequency']) 
plt.show()


plt.xticks(range(len(Category_summary['mean_monetary'])), Category_summary['new_label'])
plt.xlabel('Customer Categories')
plt.ylabel('Mean Monetary')
plt.title('Mean Monetary by Customer Categoriess')
plt.bar(range(len(Category_summary['mean_monetary'])), Category_summary['mean_monetary']) 
plt.show()
customercategory = salesgroup.groupby('new_label').agg(
    count=('Customer ID', 'count'))
customercategory.reset_index(inplace = True)
customercategory
sales_cleansed = salesgroup[['Customer ID', 'Country', 'sum_price', 'sum_quantity', 'monetary',
        'frequency', 'recency','R2', 'F2', 'M2','RFM Score2', 'new_label']]
sales_cleansed.to_csv('sales_cleansed.csv', index=False)