# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('../input/nba-games/ranking.csv', parse_dates=['STANDINGSDATE'])
df.head(30)
df.info()
#Identifikasi tanggal transaksi paling awal dan terbaru.
print(df['STANDINGSDATE'].min(), df['STANDINGSDATE'].max())
#analisis tgn 01 Maret 2020
#jumlah hari dari tanggal dihitung:

sd = dt.datetime(2020,5,1)
df['hist']=sd - df['STANDINGSDATE']
df['hist'].astype('timedelta64[D]')
df['hist']=df['hist'] / np.timedelta64(1, 'D')
df.head(50)
df=df[df['hist'] < 730]
df.info()
#Data jumlah hari transaksi terbaru, jumlah semua  transaksi dan jumlah total transaksi.

rfmTable = df.groupby('TEAM').agg({'hist': lambda x:x.min(), # Recency
                                        'G': lambda x: len(x), # Frequency
                                        'W': lambda x: x.sum()}) # Monetary Value

rfmTable.rename(columns={'hist': 'recency', 
                         'G': 'frequency', 
                         'W': 'monetary_value'}, inplace=True)
rfmTable.head(50)
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
rfmSeg.head(50)
#Total Class

rfmSeg['Total_Class'] = rfmSeg['R_Quartile'] + rfmSeg['F_Quartile'] + \
rfmSeg['M_Quartile']


rfmSeg.head()
print("Terbaik: ",len(rfmSeg[rfmSeg['RFMClass']=='414']))
print('Langganan: ',len(rfmSeg[rfmSeg['F_Quartile']==4]))
print("Juara: ",len(rfmSeg[rfmSeg['M_Quartile']==4]))
print("Tidak Aktif: ",len(rfmSeg[rfmSeg['R_Quartile']==1]))
print('Rata-rata: ', len(rfmSeg[rfmSeg['RFMClass']=='412']))
print('Biasa: ', len(rfmSeg[rfmSeg['RFMClass']=='411']))