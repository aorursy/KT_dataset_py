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
products = pd.read_csv('/kaggle/input/flipkart-products/flipkart_com-ecommerce_sample.csv')
products.head()
len(products['product_name'])
len(products['product_name'].unique()),len(products['uniq_id'].unique())
products.shape
products.info()
products.isna().sum()
products['description'][0]
from sklearn.feature_extraction.text import TfidfVectorizer
tfv = TfidfVectorizer(max_features=None,
                     strip_accents='unicode',
                     analyzer='word',
                     min_df=10,
                     token_pattern=r'\w{1,}',
                     ngram_range=(1,3),#take the combination of 1-3 different kind of words
                     stop_words='english')#removes all the unnecessary characters like the,in etc.
products['description'] = products['description'].fillna('')
#fitting the description column.
tfv_matrix = tfv.fit_transform(products['description'])#converting everythinng to sparse matrix.
tfv_matrix
tfv_matrix.shape
from sklearn.metrics.pairwise import sigmoid_kernel
sig = sigmoid_kernel(tfv_matrix,tfv_matrix)#how description of first product is related to first product and so on.
sig[0]
indices = pd.Series(products.index,index=products['product_name']).drop_duplicates()
indices.head(20)
def product_recommendation(title,sig=sig):
    indx = indices[title]
    
    #getting pairwise similarity scores
    sig_scores = list(enumerate(sig[indx]))
    
    #sorting products
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)
    
    #10 most similar products score
    sig_scores = sig_scores[1:11]
    
    #product indexes
    product_indices = [i[0] for i in sig_scores]
    
    #Top 10 most similar products
    return products['product_name'].iloc[product_indices]
n=input("Enter the name of the product: ")
print("\nTop Recommended products are: \n")
print(product_recommendation(n).unique())
