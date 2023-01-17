

import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt



df = pd.read_csv('../input/retailtransactiondata/Retail_Data_Transactions.csv', parse_dates=['trans_date'])


df.head(10)



df.info()


#Identifikasi tanggal transaksi paling awal dan terbaru.
print(df['trans_date'].min(), df['trans_date'].max())
#analisis tgn 15 April 2015
#jumlah hari dari tanggal dihitung:

sd = dt.datetime(2015,4,1)
df['hist']=sd - df['trans_date']
df['hist'].astype('timedelta64[D]')
df['hist']=df['hist'] / np.timedelta64(1, 'D')
df.head(10)
#menggunakan data selama 2 tahun saja

df=df[df['hist'] < 730]
df.info()
#Data jumlah hari transaksi terbaru, jumlah semua  transaksi dan jumlah total transaksi.

rfmTable = df.groupby('customer_id').agg({'hist': lambda x:x.min(), # Recency
                                        'customer_id': lambda x: len(x), # Frequency
                                        'tran_amount': lambda x: x.sum()}) # Monetary Value

rfmTable.rename(columns={'hist': 'recency', 
                         'customer_id': 'frequency', 
                         'tran_amount': 'monetary_value'}, inplace=True)
rfmTable.head(10)
#Membagi menjadi 4 kategori dengan menggunakan quartile

quartiles = rfmTable.quantile(q=[0.25,0.50,0.75])
print(quartiles, type(quartiles))
#untuk recency
def RClass(x,p,d):
    if x <= d[p][0.25]:
        return 4
    elif x <= d[p][0.50]:
        return 3
    elif x <= d[p][0.75]: 
        return 2
    else:
        return 1
    
#untuk frequency dan monetary
def FMClass(x,p,d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]: 
        return 3
    else:
        return 4   
rfmSeg = rfmTable
rfmSeg['R_Quartile'] = rfmSeg['recency'].apply(RClass, args=('recency',quartiles,))
rfmSeg['F_Quartile'] = rfmSeg['frequency'].apply(FMClass, args=('frequency',quartiles,))
rfmSeg['M_Quartile'] = rfmSeg['monetary_value'].apply(FMClass, args=('monetary_value',quartiles,))
rfmSeg['RFMClass'] = rfmSeg.R_Quartile.map(str) \
                            + rfmSeg.F_Quartile.map(str) \
                            + rfmSeg.M_Quartile.map(str)
rfmSeg.head(10)
#Total Class

rfmSeg['Total_Class'] = rfmSeg['R_Quartile'] + rfmSeg['F_Quartile'] + \
rfmSeg['M_Quartile']


rfmSeg.head()

print("Pelanggan Utama: ",len(rfmSeg[rfmSeg['RFMClass']=='444']))
print('Langganan: ',len(rfmSeg[rfmSeg['F_Quartile']==4]))
print("Pemborong: ",len(rfmSeg[rfmSeg['M_Quartile']==4]))
print("Beresiko: ",len(rfmSeg[rfmSeg['R_Quartile']==1]))
print('Lost: ', len(rfmSeg[rfmSeg['RFMClass']=='111']))
print('Pelanggan berpotensial: ', len(rfmSeg[rfmSeg['RFMClass']=='333']))
print('Butuh Perhatian: ', len(rfmSeg[rfmSeg['RFMClass']=='222']))
plt.title