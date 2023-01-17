

import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt



import pandas as pd
file = pd.read_csv('../input/rfm-uas/RFM.csv', parse_dates=['TrxDate'])


file.head(10)



file.info()


#Identifikasi tanggal transaksi paling awal dan terbaru.
print(file['TrxDate'].min(), file['TrxDate'].max())
#analisis tgn 15 April 2015
#jumlah hari dari tanggal dihitung:

sd = dt.datetime(2020,6,29)
file['hist']=sd - file['TrxDate']
file['hist'].astype('timedelta64[D]')
file['hist']=file['hist'] / np.timedelta64(1, 'D')
file.head(10)
#menggunakan data selama 2 tahun saja

file=file[file['hist'] < 730]
file.info()
#Data jumlah hari transaksi terbaru, jumlah semua  transaksi dan jumlah total transaksi.

rfmTable = file.groupby('CardID').agg({'hist': lambda x:x.min(), # Recency
                                        'CardID': lambda x: len(x), # Frequency
                                        'Amount': lambda x: x.sum()}) # Monetary Value

rfmTable.rename(columns={'hist': 'recency', 
                         'CardID': 'frequency', 
                         'Amount': 'monetary_value'}, inplace=True)
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

print("Pelanggan Tetap: ",len(rfmSeg[rfmSeg['RFMClass']=='444']))
print('Langganan: ',len(rfmSeg[rfmSeg['F_Quartile']==4]))
print("Pembeli banyak: ",len(rfmSeg[rfmSeg['M_Quartile']==4]))
print("Beresiko: ",len(rfmSeg[rfmSeg['R_Quartile']==1]))
print('Lost: ', len(rfmSeg[rfmSeg['RFMClass']=='111']))
print('Pelanggan berpotensial: ', len(rfmSeg[rfmSeg['RFMClass']=='333']) + len(rfmSeg[rfmSeg['RFMClass']=='444']))
print('Butuh Perhatian: ', len(rfmSeg[rfmSeg['RFMClass']=='222']) + len(rfmSeg[rfmSeg['RFMClass']=='111']))
plt.title