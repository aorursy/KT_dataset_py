import pandas as pd

import numpy as np

import sys

from itertools import combinations, groupby

from collections import Counter

from IPython.display import display
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import zipfile



dataset_orders = "order_products__train.csv"

dataset_products = 'products.csv'

dataset_departments = 'departments.csv'

dataset_aisles = 'aisles.csv'



archive_orders = zipfile.ZipFile("../input/"+dataset_orders+".zip","r")

archive_products = zipfile.ZipFile("../input/"+dataset_products+".zip","r")

archive_departments = zipfile.ZipFile("../input/"+dataset_departments+".zip","r")

archive_aisles = zipfile.ZipFile("../input/"+dataset_aisles+".zip","r")



df_order = pd.read_csv(archive_orders.open('order_products__train.csv'))

df_product = pd.read_csv(archive_products.open('products.csv'))

df_departments = pd.read_csv(archive_departments.open('departments.csv'))

df_aisles = pd.read_csv(archive_aisles.open('aisles.csv'))
name_orders = "order_products__prior.csv"

archive_orders = zipfile.ZipFile("../input/"+name_orders+".zip","r")

orders = pd.read_csv(archive_orders.open('order_products__prior.csv'))
df_orders = pd.merge(df_order, df_product, on='product_id')

df_orders.head()
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import linear_kernel

from ast import literal_eval

from sklearn.feature_extraction.text import CountVectorizer
orders.head(2)
df_product.head(2)
df_departments.head(2)
df_aisles.head(2)
df = pd.merge(pd.merge(pd.merge(df_order, 

                                          df_product, 

                                          on="product_id", how = 'inner'), 

                                   df_aisles, 

                                   on="aisle_id", how='inner'),

                      df_departments, 

                      on="department_id", how = 'inner')
df.head()
df_meta = df.groupby('product_id', as_index=False).agg(set)
df_meta.head(2)
df_meta['aisle'] = df_meta['aisle'].apply(lambda x: [i.replace(' ','') for i in x])

df_meta['department'] = df_meta['department'].apply(lambda x: [i.replace(' ','') for i in x])
df_meta.columns
df_meta['metadata'] = df_meta.apply(lambda x : ' ' + ' '.join(x['aisle']) + ' ' + ' '.join(x['department']), axis = 1)

df_meta
df_vec = df_meta.head(10000) #Reduzindo para salvar memória

df_produtos_previstos = pd.merge(df_vec[['product_id']], df_product[['product_id','product_name']], on='product_id', how='inner')



count_vec = CountVectorizer(stop_words='english')

count_vec_matrix = count_vec.fit_transform(df_vec['metadata'])

cosine_sim_matrix = cosine_similarity(count_vec_matrix, count_vec_matrix)

mapping = pd.Series(df_produtos_previstos.index, index = df_produtos_previstos['product_name'])
def recommend_products_based_on_metadata(product_input):

    

    product_index = mapping[product_input]

    

    similarity_score = list(enumerate(cosine_sim_matrix[product_index]))

    similarity_score = sorted(similarity_score, key=lambda x: x[1], reverse=True)

    

    similarity_score = similarity_score[1:10]

    product_indices = [i[0] for i in similarity_score]

    

    return (df_vec['product_name'].iloc[product_indices])
df_produtos_previstos.sample(5)
recommend_products_based_on_metadata('Over Tired and Cranky Bubble Bath')