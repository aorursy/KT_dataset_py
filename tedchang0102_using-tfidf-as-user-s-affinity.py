import numpy as np

import pandas as pd 

import scipy.sparse as ssp

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD

from sklearn.decomposition import PCA
orders = pd.read_csv("../input/orders.csv")

train_orders = pd.read_csv("../input/order_products__train.csv")

prior_orders = pd.read_csv("../input/order_products__prior.csv")

products = pd.read_csv("../input/products.csv").set_index('product_id')
prior_orders = prior_orders[prior_orders.reordered==1]

prior_ord = pd.merge(prior_orders,orders,on='order_id',how='left')

products = products.reset_index()

prior_ord.head()
prior_ord = pd.merge(prior_ord,products,on='product_id',how='left')

prior_ord[['user_id','order_id','product_id','product_name','reordered']].head()
prior_ord["product_name"] = prior_ord["product_name"].astype(str)

prior_ord = prior_ord.groupby("user_id").apply(lambda order: order['product_name'].tolist())

prior_ord = prior_ord.reset_index()

prior_ord.columns = ['user_id','product_set']

prior_ord.product_set = prior_ord.product_set.astype(str)

prior_ord.head()
tfidf = TfidfVectorizer(min_df=5, max_features=1000

                        , strip_accents='unicode',lowercase =True,

analyzer='word', token_pattern=r'\w+', use_idf=True, 

smooth_idf=True, sublinear_tf=True, stop_words = 'english')

tfidf.fit(prior_ord['product_set'])
text = tfidf.transform(prior_ord['product_set'])

svd = TruncatedSVD(n_components=2)

text = svd.fit_transform(text)

text = pd.DataFrame(text)

text.columns = ['pf_0','pf_1']

text['user_id'] = prior_ord.user_id

text.head()
import matplotlib.pyplot as plt

plt.figure(figsize=(13,13))

plt.plot(text['pf_0'].head(50),text['pf_1'].head(50),'r*',label=text['user_id'].head(50))

for row in text.head(50).itertuples():

    plt.annotate('user_'+str(row.user_id), xy=(row.pf_0,row.pf_1), 

            xytext=(row.pf_0+0.01,row.pf_1+0.01)

            

            )