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
import seaborn as sns
from matplotlib import pyplot
sns.set()
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm
tqdm.pandas()
import pandas as pd
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('display.max_colwidth', -1)
!7za e ../input/mercari-price-suggestion-challenge/train.tsv.7z
!7za e ../input/mercari-price-suggestion-challenge/test.tsv.7z
!unzip ../input/mercari-price-suggestion-challenge/test_stg2.tsv.zip -d ../working/
df_train = pd.read_table("../working/train.tsv", index_col="train_id")
df_test = pd.read_table("../working/test.tsv", index_col="test_id")
df_test2 = pd.read_table("../working/test_stg2.tsv", index_col="test_id")
print(df_train.shape)
print(df_test.shape)
print(df_test2.shape)
# Separate category types to 3 columns. 
df_train['category_name'] = df_train['category_name'].replace(np.nan, '//', regex=True)
df_train['category1'] = ""
df_train['category2'] = ""
df_train['category3'] = ""
# df_train[['category1','category2','category3']]   new_df
df_train[['category1','category2','category3']] = df_train['category_name'].str.split("/", n = 2, expand = True)
df_train['category_name'] = df_train['category_name'].replace( '//',np.nan, regex=False)
brand_average = df_train[['brand_name','price']].groupby('brand_name').mean().reset_index().rename(columns={'price':'brand_average'})
df_train = pd.merge(df_train, brand_average, on='brand_name', how='left')
category1_average = df_train[['category1','price']].groupby('category1').mean().reset_index().rename(columns={'price':'category1_average'})
df_train = pd.merge(df_train, category1_average, on='category1', how='left')
category2_average = df_train[['category2','price']].groupby('category2').mean().reset_index().rename(columns={'price':'category2_average'})
df_train = pd.merge(df_train, category2_average, on='category2', how='left')
category3_average = df_train[['category3','price']].groupby('category3').mean().reset_index().rename(columns={'price':'category3_average'})
df_train = pd.merge(df_train, category3_average, on='category3', how='left')
df_train[['brand_average','category1_average','category2_average','category3_average']] = df_train[['brand_average','category1_average','category2_average','category3_average']].fillna(999)
df_train.head()
base_cols_to_be_used = ['item_condition_id', 'shipping','brand_average','category1_average','category2_average','category3_average']
X_full = df_train[base_cols_to_be_used]
y_full = df_train['price']
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X_full, y_full)
# Separate category types to 3 columns. 
df_test2['category_name'] = df_test2['category_name'].replace(np.nan, '//', regex=True)
df_test2['category1'] = ""
df_test2['category2'] = ""
df_test2['category3'] = ""
# df_train[['category1','category2','category3']]   new_df
df_test2[['category1','category2','category3']] = df_test2['category_name'].str.split("/", n = 2, expand = True)
df_test2['category_name'] = df_test2['category_name'].replace( '//',np.nan, regex=False)
df_test2 = pd.merge(df_test2, brand_average, on='brand_name', how='left')
df_test2 = pd.merge(df_test2, category1_average, on='category1', how='left')
df_test2 = pd.merge(df_test2, category2_average, on='category2', how='left')
df_test2 = pd.merge(df_test2, category3_average, on='category3', how='left')
df_test2[['brand_average','category1_average','category2_average','category3_average']] = df_test2[['brand_average','category1_average','category2_average','category3_average']].fillna(999)
test_predict = reg.predict(df_test2[base_cols_to_be_used])
df_test2['price'] = pd.Series(test_predict)
df_test2['test_id'] = df_test2.index
df_test2[['test_id','price']].head()
df_test2[['test_id','price']].to_csv("submission.csv",index=False) #0.69