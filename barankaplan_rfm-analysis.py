import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns',None)

pd.set_option('display.float_format', lambda x : '%.0f' % x)
df_2009_2010=pd.read_excel('../input/online-retail-iixlsx/online_retail_II.xlsx')
df=df_2009_2010.copy()
df.head()
#df['Description'].nunique
#df.groupby('Description').agg(['count'])
df['Description'].value_counts().head(10)
df.groupby('Description').agg({'Quantity':sum}).head()
found = df[df['Invoice'].str.startswith('C', na=False)]
print(found.count())
df.groupby('Description').agg({'Quantity':'sum'}).sort_values(by='Quantity',ascending=False)
df['Invoice'].nunique() #az degil mi
df['TotalPrice']=df['Quantity']*df['Price']
df.head()
df.groupby('Invoice').agg({'TotalPrice' : 'mean'}).head()
# her fatura unique ise nasil mean oluyor
df.sort_values(by='Price',ascending=False).head()
#df.groupby(['Country','Invoice'])
#df['Country'].value_counts()
#df.groupby('Country').
#df.groupby('Invoice')['Country'].apply(lambda x: x.sum()).head(5)
#df.groupby([]'Invoice').sort_values(by='Quantity',ascending=False)
df.groupby('Country')['Invoice'].count()

df.groupby('Country').agg({'TotalPrice' : 'sum'}).sort_values(by='TotalPrice',ascending=False)
#en cok iade alan ürün
found = df[df['Invoice'].str.startswith('C', na=False)]
found.sort_values(by='Quantity',ascending=True)
df.isnull().sum()
df.dropna(inplace=True)
df.isnull().sum()
df.describe([0.01,0.05,0.10,0.25,0.50,0.75,0.90,0.95,0.99]).T
df.info()
df['InvoiceDate'].min()
df['InvoiceDate'].max()
import datetime as dt
today_date=dt.datetime(2011,12,9)
df.groupby('Customer ID').agg({'InvoiceDate':'max'}).head()
df['Customer ID']=df['Customer ID'].astype(int)
today_date-df.groupby('Customer ID').agg({'InvoiceDate':'max'}).head()
temp_df=(today_date-df.groupby('Customer ID').agg({'InvoiceDate':'max'}))
temp_df.rename(columns={'InvoiceDate':'Recency'},inplace=True)
temp_df.head()
recency_df=temp_df['Recency'].apply(lambda x :x.days)
recency_df.head()
temp_df=df.groupby(['Customer ID','Invoice']).agg({'Invoice' : 'count'})
temp_df.head()
freq_df=temp_df.groupby('Customer ID').agg({'Invoice' : 'count'})
freq_df.rename(columns={'Invoice':'Frequency'}, inplace=True)
freq_df.head()
monetary_df=df.groupby('Customer ID').agg({'TotalPrice':'sum'})
monetary_df.head()
monetary_df.rename(columns={'TotalPrice':'Monetary'}, inplace=True)
print(recency_df.shape,freq_df.shape,monetary_df.shape)
rfm=pd.concat([recency_df,freq_df,monetary_df], axis=1)
rfm.head()
rfm['RecencyScore']=pd.qcut(rfm['Recency'],5,labels=[5,4,3,2,1])
rfm['FrequencyScore']=pd.qcut(rfm['Frequency'].rank(method='first'),5,labels=[1,2,3,4,5])
rfm['MonetaryScore']=pd.qcut(rfm['Monetary'],5,labels=[1,2,3,4,5])
rfm.head()
(rfm['RecencyScore'].astype(str)+rfm['FrequencyScore'].astype(str)+rfm['MonetaryScore'].astype(str)).head()
rfm['RFM_SCORE']=rfm['RecencyScore'].astype(str)+rfm['FrequencyScore'].astype(str)+rfm['MonetaryScore'].astype(str)
rfm.head()

#pd.set_option('display.max_rows',None)
rfm[rfm['RFM_SCORE']>='300'].head().sort_values(by='RFM_SCORE',ascending=False)

seg_map={r'[1-2][1-2]': 'Hibernating',
        r'[1-2][3-4]': 'At Risk',
        r'[1-2]5': 'Can\'t loose',
        r'3[1-2]': 'About to Sleep',
        r'33' : 'Need Attention',
        r'[3-4][4-5]': 'Loyal Customers',
        r'41' : 'Promising',
        r'51' : 'New Customers', 
        r'[4-5][2-3]': 'Potential Loyalists',
        r'5[4-5]':'Champions'
          
        }
rfm['Segment']=rfm['RecencyScore'].astype(str)+rfm['FrequencyScore'].astype(str)
rfm['Segment']=rfm['Segment'].replace(seg_map,regex=True)
rfm.head()

rfm[['Segment','Recency','Frequency','Monetary']].groupby('Segment').agg(['mean' , 'count'])
rfm[rfm['Segment']=='Can\'t loose']
segments_counts = rfm['Segment'].value_counts().sort_values(ascending=True)

fig, ax = plt.subplots()

bars = ax.barh(range(len(segments_counts)),
              segments_counts,
              color='silver')
ax.set_frame_on(False)
ax.tick_params(left=False,
               bottom=False,
               labelbottom=False)
ax.set_yticks(range(len(segments_counts)))
ax.set_yticklabels(segments_counts.index)

for i, bar in enumerate(bars):
        value = bar.get_width()
        if segments_counts.index[i] in ['Can\'t loose']:
            bar.set_color('firebrick')
        ax.text(value,
                bar.get_y() + bar.get_height()/2,
                '{:,} ({:}%)'.format(int(value),
                                   int(value*100/segments_counts.sum())),
                va='center',
                ha='left'
               )

plt.show()
