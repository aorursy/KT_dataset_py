# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import google

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
customers = pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
customers.head()
customers.info()
customers.isnull().sum()
customers_null = customers
for col in customers_null.columns:
    customers_null.loc[customers_null.sample(frac=0.1).index,col] = np.nan
customers_null.head()
customers_null.info()
customers_null.isnull().sum()

customers_null_1 = customers_null
customers_null_2 = customers_null
customers_null_1.dropna()
customers_null_2.fillna(0)
customers_null.describe()
customers_null_3 = customers_null
customers_null_3.fillna(customers_null_2.mean())
boxplot = customers_null_3.boxplot(column=['Age','Annual Income (k$)','Spending Score (1-100)'])
from scipy  import stats
z = np.abs(stats.zscore(customers['Annual Income (k$)'].values))
threshold = 2
result = np.where(z > threshold)
df_salario_oulier = customers_null_3.iloc[result[0]]
df_salario_oulier
sns.countplot(x='Gender', data=customers)
plt.title('Distribuição de cliente por gênero')
customers.hist('Age', bins=35)
plt.title('Distribuição de clientes por idade')
plt.xlabel('Idade')
cat_df_customers = customers.select_dtypes(include=['object'])
cat_df_customers.head()

replace_map = {'Gender': {'Male':1,"Female":2}}
labels = cat_df_customers['Gender'].astype('category').cat.categories.tolist()
replace_map_comp = {'Gender': {k: v for k, v in zip(labels,list(range(1,len(labels)+1)))}}
replace_map_comp
cat_df_customers_replace = pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
cat_df_customers_replace.replace(replace_map_comp,inplace=True)
cat_df_customers_replace.head()
customers = pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
cat_df_customers_lc=customers
cat_df_customers_lc['Gender'] = pd.Categorical(cat_df_customers_lc['Gender'])
cat_df_customers_lc.dtypes
cat_df_customers_lc['Gender'] = cat_df_customers_lc['Gender'].cat.codes
cat_df_customers_lc.head()
