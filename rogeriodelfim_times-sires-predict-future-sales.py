import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import catboost

import matplotlib.pyplot as plt

# from sklearn.preprocessing import train_test_split as split

import seaborn as sns





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
path = '/kaggle/input/competitive-data-science-predict-future-sales/'
df_train     = pd.read_csv(path + 'sales_train.csv')

df_test      = pd.read_csv(path + 'test.csv')

df_shopes    = pd.read_csv(path + 'shops.csv')

df_items     = pd.read_csv(path + 'items.csv')

df_categorie = pd.read_csv(path + 'item_categories.csv')

df_submission= pd.read_csv(path + 'sample_submission.csv') 
df_train.head()
df_train.info()
df_test.head()
df_test.info()
df_categorie.head()
df_categorie.info()
df_items.head()
df_items.info()
df_shopes.head()
plt.figure(figsize = (10,4))

plt.xlim(-100, 3000)

sns.boxplot( x = df_train.item_cnt_day )

plt.figure(figsize = (10,4) )



plt.figure( figsize = (10,4) )

plt.xlim(df_train.item_price.min(), df_train.item_price.max())

sns.boxplot( x = df_train.item_price )

plt.show()
df_train["item_cnt_day"].plot(figsize=(10, 6));