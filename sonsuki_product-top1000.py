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
import pandas as pd

from pandas import DataFrame

import numpy as np
train = pd.read_csv("../input/instacart-market-basket-analysis/order_products__train.csv")

prior = pd.read_csv("../input/instacart-market-basket-analysis/order_products__prior.csv")

orders = pd.read_csv("../input/instacart-market-basket-analysis/orders.csv")

products = pd.read_csv("../input/instacart-market-basket-analysis/products.csv")

aisles = pd.read_csv("../input/instacart-market-basket-analysis/aisles.csv")

departments = pd.read_csv("../input/instacart-market-basket-analysis/departments.csv")
prior = pd.concat([train, prior])
#prior[:10]
prior1 = prior.join(products.set_index('product_id'), on='product_id')
from collections import Counter as cc



pro_name = cc(prior1['product_name'])

A = pro_name.most_common()[:1000]

#print(A)
A = DataFrame(A)

A.columns = ["product_name","product_count"]
#print(A)
top10 = A.join(prior1.set_index('product_name'), on='product_name')
len(top10)
All_top10 = pd.merge(top10, orders, how='left')
#All_top10
All_top10['user_order'] =All_top10['user_id'] + All_top10['order_number'] * 0.1
All_top10.drop(['eval_set', 'product_count', 'product_id', 'user_id', 'order_number'], axis='columns', inplace=True)

All_top10.head()
group_top10 = All_top10["product_name"].groupby(All_top10['user_order'])

group_top10 = pd.DataFrame(group_top10)
group_top10.columns = ['user_order','product_list']
group_top10.head()
product = [list(s) for s in group_top10.product_list]
#product
len(product)
len(group_top10)
from gensim.models.word2vec import Word2Vec
%%time

model = Word2Vec(product)
model.init_sims(replace=True)
model.wv.similarity('Banana', 'Organic Whole Milk')
model.wv.similarity('Red Mango', 'Honeydew Melon')
model.wv.similarity('Bag of Organic Bananas', 'Organic Whole Milk')
model.wv.most_similar("Banana")
model.wv.most_similar("Organic Avocado")
model.wv.most_similar("Organic Hass Avocado")
model.wv.most_similar("Bag of Organic Bananas")
model.wv.most_similar("Organic Gala Apples")
model.wv.most_similar("Apple Honeycrisp Organic")
model.wv.most_similar("Organic Strawberries")
model.wv.most_similar("Seedless Red Grapes")
model.wv.most_similar("Granny Smith Apples")
model.wv.most_similar("Fat Free Milk")
model.wv.most_similar("Original Orange Juice")
model.wv.most_similar("Honey Nut Cheerios")
model.wv.most_similar("Lemonade")
model.wv.most_similar("Organic Blueberries")
model.wv.most_similar("Blueberries")
model.wv.most_similar("Granny Smith Apples")
model.wv.most_similar("Organic Half & Half")
model.wv.most_similar("Half & Half")
model.wv.most_similar("Organic Fuji Apple")
model.wv.most_similar("Honeycrisp Apple")
model.wv.most_similar("Strawberries")
model.wv.most_similar("Organic Baby Spinach")
model.wv.most_similar("Limes")
model.wv.most_similar("Extra Virgin Olive Oil")
model.wv.most_similar("Asparagus")
model.wv.most_similar("Organic Whole Milk")
model.wv.most_similar("Organic Reduced Fat Milk")
model.wv.most_similar("Whipped Cream Cheese")
model.wv.most_similar("Organic Baby Spinach")
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt



def tsne_plot(model):

    labels = []

    tokens = []



    for word in model.wv.vocab:

        tokens.append(model[word])

        labels.append(word)

    

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)

    new_values = tsne_model.fit_transform(tokens)



    x = []

    y = []

    for value in new_values:

        x.append(value[0])

        y.append(value[1])

        

    plt.figure(figsize=(128, 128)) 

    for i in range(len(x)):

        plt.scatter(x[i],y[i])

        plt.annotate(labels[i],

                     xy=(x[i], y[i]),

                     xytext=(5, 2),

                     textcoords='offset points',

                     ha='right',

                     va='bottom')

    plt.show()
#tsne_plot(model)
model.save('product_model_top1000.model')