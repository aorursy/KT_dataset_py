!pip install pyforest 

#Install this and thank me later
from pyforest import *
df = pd.read_excel('/kaggle/input/online-retail-store-data-from-uci-ml-repo/Online Retail.xlsx', encoding = 'unicode_escape', parse_dates = ['InvoiceDate'])
#no need to sit and try to remember all the libraries you are going to need

active_imports()
lazy_imports()
df.head()
df.info()
df.describe(include = 'all')
df.Country.value_counts()
#let's focus on the customers from France. 



france_customers = df[df.Country == 'France']
france_customers.head()
france_customers.info()
france_customers.describe(include = 'all')
negative_quantity = france_customers.query('Quantity < 0')
negative_quantity
# # %load Data Dictionary.txt

# Attribute Information:



# InvoiceNo: Invoice number. Nominal, a 6-digit integral number uniquely assigned to each transaction. If this code starts with letter 'c', it indicates a cancellation.

# StockCode: Product (item) code. Nominal, a 5-digit integral number uniquely assigned to each distinct product.

# Description: Product (item) name. Nominal.

# Quantity: The quantities of each product (item) per transaction. Numeric.

# InvoiceDate: Invice Date and time. Numeric, the day and time when each transaction was generated.

# UnitPrice: Unit price. Numeric, Product price per unit in sterling.

# CustomerID: Customer number. Nominal, a 5-digit integral number uniquely assigned to each customer.

# Country: Country name. Nominal, the name of the country where each customer resides.
columns_to_drop = negative_quantity.index.tolist()
france_customers.drop(index = columns_to_drop, inplace = True)
france_customers.shape
france_customers['Quantity'].describe()
france_customers['InvoiceNo'] = france_customers['InvoiceNo'].astype(str)
france_customers.InvoiceNo.str.contains('C').sum()
threshold_date = dt.datetime(2011,12,10)
france_customers['Year'] = france_customers['InvoiceDate'].dt.year
france_customers.head()
#we only want to work with data for the one year so let us choose the year 2011. 

france_customers.Year.value_counts()
france_customers = france_customers[france_customers.Year == 2011]
france_customers.shape
france_customers.isnull().sum()
france_customers[france_customers.CustomerID.isnull()]
france_customers.dropna(inplace = True)
france_customers.isnull().sum()
rfm_data = france_customers.copy()
rfm_data.head()
%who
rfm_data['Days_since_purchase'] = threshold_date - rfm_data['InvoiceDate']
rfm_data['Days_since_purchase'] = rfm_data['Days_since_purchase'].dt.days
rfm_data.head()
rfm_data['Total_amount_spent'] = rfm_data['Quantity'] * rfm_data['UnitPrice']
rfm_data.head()
group_data = rfm_data.groupby('CustomerID')
columns = ['Recency', 'Frequency', 'Monetary']
rfm = pd.DataFrame(columns = columns)
rfm
rfm['Recency'] = group_data['Days_since_purchase'].max()
rfm
rfm['Frequency'] = group_data['InvoiceNo'].size()
rfm
rfm['Monetary'] = group_data['Total_amount_spent'].sum()
rfm
rfm.shape
rfm_quantiles = rfm.quantile(q=[0.25,0.5,0.75])
rfm_quantiles = rfm_quantiles.to_dict()
rfm_quantiles
plt.style.available
plt.style.use('fivethirtyeight')

plt.figure(figsize = (8,8))

sns.distplot(rfm['Frequency'])
rfm_quantiles
rfm['Recency'] = rfm['Recency'].astype(int)
def recency_score(value,d,l):

#d is the dictionary

#l is key label



    if value <= d[l][0.25]:

        return 1

    elif value <= d[l][0.5]:

        return 2

    elif value <= d[l][0.75]:

        return 3

    else:

        return 4

    
def frequency_score(value,d,l):

#d is the dictionary

#l is key label

# be careful with assigning the scores here. Since a higher frequency means better for our business, we need to reverse the

# scoring criteria



    if value <= d[l][0.25]:

        return 4

    elif value <= d[l][0.5]:

        return 3

    elif value <= d[l][0.75]:

        return 2

    else:

        return 1
#this is a redundant step.only for the purpose of this notebook am I repeating the formula for monetary score otherwise the same

#formula for frequency_score can be used





def monetary_score(value,d,l):

#d is the dictionary

#l is key label

# be careful with assigning the scores here. Since a higher frequency means better for our business, we need to reverse the

# scoring criteria



    if value <= d[l][0.25]:

        return 4

    elif value <= d[l][0.5]:

        return 3

    elif value <= d[l][0.75]:

        return 2

    else:

        return 1
rfm['Recency_score'] = rfm['Recency'].apply(func = recency_score, args = (rfm_quantiles, 'Recency'))
rfm.head()
rfm['Frequency_score'] = rfm['Frequency'].apply(func = frequency_score, args = (rfm_quantiles, 'Frequency'))
rfm.head()
rfm['Monetary_score'] = rfm['Monetary'].apply(func = frequency_score, args = (rfm_quantiles, 'Monetary'))
rfm.head()
rfm['RFM_Score'] = rfm[['Recency_score','Frequency_score','Monetary_score']].sum(axis = 1)
rfm.head()
plt.style.use('default')

plt.figure(figsize = (7,7))

rfm['RFM_Score'].plot.hist()

plt.axvline(rfm['RFM_Score'].mean(), color = 'orange', label = 'Mean RFM Score')

plt.legend()

plt.show()
rfm['Customer_Class'] = pd.cut(rfm['RFM_Score'],4, labels = ['Platinum','Gold','Silver','Bronze'])
rfm.head()
rfm.Customer_Class.value_counts()