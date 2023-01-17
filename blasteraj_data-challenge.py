# Lets import packages
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
p = sns.color_palette()
import gc
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/data1.csv')
print('Number of rows: {}, Number of columns: {}'.format(*df.shape))
df.head()
df.info()
print('Number of rows: {}, in which \'Quantity\' is taking negative values'.format(len(df[df['Quantity']<0])))
df[df['Quantity']<0][:10]
#Printing those columns.
print(df[df['UnitPrice']<0])
# Deleting those columns.
df.drop(df[df['UnitPrice']<0].index,inplace=True)
df['Overall_Price']=df['Quantity']*df['UnitPrice']
df['word_count'] = df['Description'].apply(lambda x: len(str(x).split(" ")))
df['char_count'] = df['Description'].str.len() ## this also includes spaces
df.head()
temp_sum_price = df['Overall_Price'].groupby(df['StockCode']).sum().reset_index().sort_values(by=['Overall_Price'],ascending=False)
temp_sum_price.columns = ['StockCode','Overall_Price_Sum']
#temp_sum_price = df['Overall_Price'].groupby(df['StockCode']).sum().reset_index()
temp_count_price = df['Overall_Price'].groupby(df['StockCode']).count().reset_index()
temp_count_price.columns = ['StockCode','Overall_Price_Count']
temp_final = pd.merge(temp_sum_price,temp_count_price,on='StockCode')
temp_final.head()
del temp_sum_price
del temp_count_price
gc.collect()
stk_cds_to_ana = temp_final['StockCode'].values
for i in range(0,10):
    temp=df[df['StockCode']==stk_cds_to_ana[i]]
    temp['InvoiceDate']=pd.to_datetime(temp['InvoiceDate'])
    temp.set_index('InvoiceDate',inplace=True)
    temp[['Quantity','Overall_Price']].plot(figsize=(20,7), linewidth=5, fontsize=10, colors=['k','r'])
    plt.xlabel('Year', fontsize=15)
    plt.ylabel('Magnitude', fontsize=15)
    plt.title('StockCode:'+stk_cds_to_ana[i],fontsize=15)
    del temp
    gc.collect()
    
    
for i in range(-10,0):
    temp=df[df['StockCode']==stk_cds_to_ana[i]]
    temp['InvoiceDate']=pd.to_datetime(temp['InvoiceDate'])
    temp.set_index('InvoiceDate',inplace=True)
    temp[['Quantity','Overall_Price']].plot(figsize=(20,7), linewidth=5, fontsize=10, colors=['k','r'])
    plt.xlabel('Year', fontsize=15)
    plt.ylabel('Magnitude', fontsize=15)
    plt.title('StockCode:'+stk_cds_to_ana[i],fontsize=15)
    del temp
    gc.collect()
print('Correlation value is ',df['Quantity'].corr(df['UnitPrice']))
temp_word_count = df['Overall_Price'].groupby(df['word_count']).mean().reset_index()
temp_word_count.columns = ['word_count','Overall_Price_Mean']
temp_word_count.plot(x='word_count',y='Overall_Price_Mean',color = 'r',figsize=(10,10))
plt.title('Overall_Mean_Price v/s Description_Word_Count',fontsize=15)
plt.xlabel('Description_Word_Count',fontsize=15)
plt.ylabel('Overall_Mean_Price',fontsize=15)
del temp_word_count
gc.collect()
temp_char_count = df['Overall_Price'].groupby(df['char_count']).mean().reset_index()
temp_char_count.columns = ['char_count','Overall_Price_Mean']
temp_char_count.plot(x='char_count',y='Overall_Price_Mean',color = 'r',figsize=(10,10))
plt.title('Overall_mean_price v/s Description_Char_Count',fontsize=15)
plt.xlabel('Description_Char_count',fontsize=15)
plt.ylabel('Overall_Mean_Price',fontsize=15)
del temp_char_count
gc.collect()