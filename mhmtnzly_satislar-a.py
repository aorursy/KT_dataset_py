# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from mlxtend.frequent_patterns import apriori, association_rules 

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
pd.options.display.max_columns

pd.options.display.max_rows = 100
store_data = pd.read_csv('/kaggle/input/satislar.csv',sep=';',low_memory=False, header=None)
store_data.head()
store_data.columns = ['BranchId', 'PosId', 'InvoiceDate', 'InvoiceNo','StockCode','Line','Quantity','CategoryCode','CategoryName']

store_data.head()
for i in range(len(store_data.CategoryName)):

    if store_data.CategoryName[i][-1]==",":

        store_data.CategoryName[i]=store_data.CategoryName[i][:-1]

    if store_data.CategoryName[i][-1]==",":

        store_data.CategoryName[i]=store_data.CategoryName[i][:-1]
store_data.head()
store_data.dropna(subset= ["CategoryCode"],inplace= True)

store_data.shape
store_data["CategoryName"].nunique()
store_data["BranchId"].value_counts()
store_data.info()
store_data["InvoiceNo"].nunique()
store_data.describe().T
# Stripping extra spaces in the description 

store_data['CategoryName'] = store_data['CategoryName'].str.strip() 
# Dropping the rows without any invoice number 

store_data.dropna(subset =['InvoiceNo'], inplace = True) 
store_data
maskBrachId = store_data["BranchId"] == 4010

df= store_data[maskBrachId]
df
df.dropna(subset= ["CategoryName"],inplace= True)
df
pd.DataFrame(df["CategoryName"].value_counts(normalize=True)).head(10)
df["BranchId"].value_counts()
df["PosId"].value_counts()
df.info()
df['Quantity'] = [x.replace(',', '.') for x in df['Quantity']]
df["Quantity"] = df["Quantity"].astype("float")
df["Quantity"] = df["Quantity"].astype("int")
df.info()
branch_order = (df.groupby(['InvoiceNo', 'CategoryName'])['Quantity'].sum().unstack().reset_index().fillna(0).set_index('InvoiceNo')) 
branch_order.head(20)
# Encoding the datasets 

branch_encoded = branch_order.applymap(lambda x: 0 if x<=0 else 1) 

basket_branch = branch_encoded 
frq_items = apriori(basket_branch, min_support = 0.0005, use_colnames = True)
frq_items
from mlxtend.frequent_patterns import association_rules

ass_rules = association_rules(frq_items, metric ="confidence", min_threshold = 0.20) 

ass_rules.head() 
rules = association_rules(frq_items, metric="lift", min_threshold=1.2)

rules
rules2 = rules.sort_values(['lift','confidence'], ascending =[False, False]) 

rules2.head(100)