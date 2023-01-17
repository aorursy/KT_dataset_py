#installation of libraries

import pandas as pd

import numpy as np

import seaborn as sns



#to display all columns and rows:

pd.set_option('display.max_columns', None); pd.set_option('display.max_rows', None);



#we determined how many numbers to show after comma

pd.set_option('display.float_format', lambda x: '%.0f' % x)

import matplotlib.pyplot as plt
#calling the dataset

df = pd.read_csv("../input/online-retail-ii-uci/online_retail_II.csv")
#selection of the first 5 observations

df.head() 
#ranking of the most ordered products

df.groupby("Description").agg({"Quantity":"sum"}).sort_values("Quantity", ascending = False).head()
#how many invoices are there in the data set

df["Invoice"].nunique()
#which are the most expensive products?

df.sort_values("Price", ascending = False).head()
#top 5 countries with the highest number of orders

df["Country"].value_counts().head()
#total spending was added as a column

df['TotalPrice'] = df['Price']*df['Quantity']
#which countries did we get the most income from

df.groupby("Country").agg({"TotalPrice":"sum"}).sort_values("TotalPrice", ascending = False).head()
df["InvoiceDate"].min() #oldest shopping date
df["InvoiceDate"].max() #newest shopping date
#to make the assessment easier, today's date is set as January 1, 2012.  

today = pd.datetime(2012,1,1) 

today
#changing the data type of the order date

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
#taking values greater than 0, this will be easier in terms of evaluation

df = df[df['Quantity'] > 0]

df = df[df['TotalPrice'] > 0]
df.dropna(inplace = True) #removal of observation units with missing data from df
df.shape #size information
df.describe([0.01,0.05,0.10,0.25,0.50,0.75,0.90,0.95, 0.99]).T

#explanatory statistics values of the observation units corresponding to the specified percentages

#processing according to numerical variables
df.head()
df.info() 

#dataframe's index dtype and column dtypes, non-null values and memory usage information
# finding Recency and Monetary values.

df_x = df.groupby('Customer ID').agg({'TotalPrice': lambda x: x.sum(), #monetary value

                                        'InvoiceDate': lambda x: (today - x.max()).days}) #recency value

#x.max()).days; last shopping date of customers
df_y = df.groupby(['Customer ID','Invoice']).agg({'TotalPrice': lambda x: x.sum()})

df_z = df_y.groupby('Customer ID').agg({'TotalPrice': lambda x: len(x)}) 

#finding the frequency value per capita
#creating the RFM table

rfm_table= pd.merge(df_x,df_z, on='Customer ID')
#determination of column names

rfm_table.rename(columns= {'InvoiceDate': 'Recency',

                          'TotalPrice_y': 'Frequency',

                          'TotalPrice_x': 'Monetary'}, inplace= True)
rfm_table.head()
#RFM score values 

rfm_table['RecencyScore'] = pd.qcut(rfm_table['Recency'],5,labels=[5,4,3,2,1])

rfm_table['FrequencyScore'] = pd.qcut(rfm_table['Frequency'].rank(method="first"),5,labels=[1,2,3,4,5])

rfm_table['MonetaryScore'] = pd.qcut(rfm_table['Monetary'],5,labels=[1,2,3,4,5])
rfm_table.head()
#RFM score values are combined side by side in str format

(rfm_table['RecencyScore'].astype(str) + 

 rfm_table['FrequencyScore'].astype(str) + 

 rfm_table['MonetaryScore'].astype(str)).head()
#calculation of the RFM score

rfm_table["RFM_SCORE"] = rfm_table['RecencyScore'].astype(str) + rfm_table['FrequencyScore'].astype(str) + rfm_table['MonetaryScore'].astype(str)
rfm_table.head()
#transposition of the RFM table. This makes it easier to evaluate.

rfm_table.describe().T
#customers with RFM Score 555

rfm_table[rfm_table["RFM_SCORE"] == "555"].head()
#customers with RFM Score 111

rfm_table[rfm_table["RFM_SCORE"] == "111"].head()
#segmenting of customers according to RecencyScore and FrequencyScore values

seg_map = {

    r'[1-2][1-2]': 'Hibernating',

    r'[1-2][3-4]': 'At Risk',

    r'[1-2]5': 'Can\'t Lose',

    r'3[1-2]': 'About to Sleep',

    r'33': 'Need Attention',

    r'[3-4][4-5]': 'Loyal Customers',

    r'41': 'Promising',

    r'51': 'New Customers',

    r'[4-5][2-3]': 'Potential Loyalists',

    r'5[4-5]': 'Champions'

}
#creation of segment variable

rfm_table['Segment'] = rfm_table['RecencyScore'].astype(str) + rfm_table['FrequencyScore'].astype(str)

rfm_table['Segment'] = rfm_table['Segment'].replace(seg_map, regex=True)
rfm_table.head()
rfm_table[["Segment", "Recency","Frequency","Monetary"]].groupby("Segment").agg(["mean","count"])