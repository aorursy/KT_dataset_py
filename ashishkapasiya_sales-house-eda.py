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
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
from glob import glob
%matplotlib inline
df0 = pd.read_csv('../input/sales-prediction/Sales_January_2019.csv')
df1 = pd.read_csv('../input/sales-prediction/Sales_February_2019.csv')
df2 = pd.read_csv('../input/sales-prediction/Sales_March_2019.csv')
df3 = pd.read_csv('../input/sales-prediction/Sales_April_2019.csv')
df4 = pd.read_csv('../input/sales-prediction/Sales_May_2019.csv')
df5 = pd.read_csv('../input/sales-prediction/Sales_June_2019.csv')
df6 = pd.read_csv('../input/sales-prediction/Sales_July_2019.csv')
df7 = pd.read_csv('../input/sales-prediction/Sales_August_2019.csv')
df8 = pd.read_csv('../input/sales-prediction/Sales_September_2019.csv')
df9 = pd.read_csv('../input/sales-prediction/Sales_October_2019.csv')
df10 = pd.read_csv('../input/sales-prediction/Sales_November_2019.csv')
df11 = pd.read_csv('../input/sales-prediction/Sales_December_2019.csv')

df_new = pd.concat([df0,df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11])
df_new.to_csv('df.csv',index=False)
df = pd.read_csv('df.csv')
df.head()
df.shape
df.info()
df.isnull().sum()
df.dropna(how='any',inplace=True)
print(df[pd.to_numeric(df['Price Each'], errors='coerce').isnull()])
df = df[df['Price Each'].str[0:10] != 'Price Each']
df['Order Date']=pd.to_datetime(df['Order Date'],errors='coerce')
df['Price Each']=pd.to_numeric(df['Price Each'],errors='coerce')
df['Quantity Ordered']=pd.to_numeric(df['Quantity Ordered'],errors='coerce')
df['Order ID']=pd.to_numeric(df['Order ID'],errors='coerce')
df['Sales']=df['Quantity Ordered']*df['Price Each']
df['City']=df['Purchase Address'].apply(lambda x: x.split(',')[1]+' '+x.split(',')[2])
df['Month']= df['Order Date'].dt.month
df['Year']= df['Order Date'].dt.year
df.groupby('Month').Sales.sum().nlargest()
df.groupby('City').Sales.sum().nlargest()
