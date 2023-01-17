# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df=pd.read_csv("/kaggle/input/bitcoin-historical-data/bitstampUSD_1-min_data_2012-01-01_to_2019-08-12.csv")
df.head()
df=pd.read_csv("/kaggle/input/bitcoin-historical-data/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv")
df.head()
df=pd.read_csv("/kaggle/input/bitcoin-historical-data/bitstampUSD_1-min_data_2012-01-01_to_2019-08-12.csv")
df.tail()

df=pd.read_csv("/kaggle/input/bitcoin-historical-data/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv")
df.tail()

df.columns 
df.shape
df.info
print(df['High'].value_counts(dropna =False))
df.describe()
df.boxplot(column='High', by='Low')
plt.show()
df_new=df.head()
df_new
melted=pd.melt(frame=df_new, id_vars='Open', value_vars=['High','Low'])
melted
melted.pivot(index='Open', columns='variable', values='value') #veri şeklinden dolayı birleştirilemiyor-ama kod dogru
data1=df.head()
data2=df.tail()
conc_data_row=pd.concat([data1,data2],axis=0, ignore_index=True) #ignore_index sayesinde baştan ve sonran sayılar sıralı olarak görünüyor
conc_data_row

data1=df['High'].head()
data2=df['Low'].head()
conc_data_col=pd.concat([data1,data2], axis=1)
conc_data_col
df.dtypes
df['Open']=df['Open'].astype('int') #veri türünü integer'a cevirdik
df['Low']=df['Low'].astype('category') #veri türünü category'e cevirdik
df.dtypes #ekrana yazdırarak kontrol gerçekleştirdik
df.info()
df['High'].value_counts(dropna=False) #DEGERİ nan olanları da göstermeyi saglar
