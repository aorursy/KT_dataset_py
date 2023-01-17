# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

pd.options.display.max_columns
pd.options.display.max_rows = 100
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
satislar = pd.read_csv('/kaggle/input/satislar.csv',sep=';',low_memory=False, header=None)
satislar.head()
satislar.columns = ['BranchId', 'PosId', 'InvoiceDate', 'InvoiceNo','StockCode','Line','Quantity','CategoryCode','CategoryName']
satislar.head(20)
df = satislar.copy()
df.drop(columns= ['PosId', "BranchId", "StockCode", "Line"], inplace=True) # işimize yaramayan column'ları attık.
df.head()
df.isna().sum()
df.dropna(inplace=True) # nan olan verileri attık 
df.shape
df.tail(20)
df['CategoryName'] = df['CategoryName'].apply(lambda x: x.strip(",")) # CategoryName sonundaki virgülleri attık
df
df["InvoiceNo"].nunique() # toplam kaç fatura kesilmiş
df['CategoryName'].value_counts().head(10)
df.groupby(["InvoiceNo", 'CategoryName'])['CategoryName'].count() # Hangi faturada, neler alınmış
items = (df['CategoryName'].unique())
items[:10]
branch_order = (df
          .groupby(['InvoiceNo', 'CategoryName'])['Quantity'] 
          .sum().unstack().reset_index().fillna(0) 
          .set_index('InvoiceNo'))
branch_order.head(10)
branch_order.isna().sum()
encoded = branch_order.applymap(lambda x: 1 if x != 0 else 0)
freq_items = apriori(encoded, min_support=0.00, use_colnames=True, verbose=1)
freq_items.sort_values('support', ascending=False)
association_rules(freq_items, metric = 'confidence', min_threshold=0.4).sort_values(['support','confidence'], ascending=[False,False])
satislar["BranchId"].nunique() # Şubeymiş
satislar["InvoiceNo"].nunique()
satislar.groupby(['BranchId'])['InvoiceNo'].count()[:10] # Hangi şubede kaç fatura kesildi?
