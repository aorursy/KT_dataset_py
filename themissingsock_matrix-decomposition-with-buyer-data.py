import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.sparse import csr_matrix

from collections import Counter
order_prior = pd.read_csv("../input/order_products__prior.csv")

orders = pd.read_csv("../input/orders.csv")

train = pd.read_csv("../input/order_products__train.csv")

products = pd.read_csv("../input/products.csv")
test = orders[orders['eval_set']=='test']

prior = orders[orders['eval_set']=='prior']

prior.tail()
test = pd.merge(test, prior[['user_id', 'order_id', 'order_number']], how = 'left', on = 'user_id')

print(test.shape)

test.head()
test = pd.merge(test, order_prior, left_on = 'order_id_y', right_on = 'order_id')

test['new_order_id'] = test['order_id_x']

test['prior_order_id'] = test['order_id_y']

test = test.drop(['order_id_x', 'order_id_y'], axis = 1)

del [orders, order_prior, train]

lookup_aisles = test.product_id.map(products.set_index('product_id')['department_id'])

test['aisles'] = lookup_aisles
test.head()
product_list = test[test['reordered']==1].groupby(['user_id', 'order_number_x', 'new_order_id']).agg({'product_id': lambda x: tuple(x),

                                                                                                      'aisles': lambda x: tuple(x)})

product_list = pd.DataFrame(product_list.reset_index())

product_list['num_products_reordered'] = product_list.product_id.apply(len)

product_list.head(15)
test[(test['user_id'] == 4)]
indptr = [0]

indices = []

data = []

column_position = {}

# input must be a list of lists

for order in product_list['product_id']:

    for product in order:

        index = column_position.setdefault(product, len(column_position))

        indices.append(index)

        data.append(1)

    indptr.append(len(indices))

    

prod_matrix = csr_matrix((data, indices, indptr), dtype=int)

#del(test)
prod_matrix.shape
from sklearn.decomposition import NMF

from sklearn.preprocessing import normalize



nmf = NMF(n_components = 50, random_state = 42)

model = nmf.fit(prod_matrix)

H = model.components_

model.components_.shape
W = model.transform(prod_matrix)

user_data = pd.DataFrame(normalize(W), index = product_list['user_id'])

idx = user_data.dot(user_data.iloc[0]).nlargest(5).index

user_data.dot(user_data.iloc[0]).nlargest(5)
W.shape
H.shape
#tmp = np.dot(W[25], H).argmax()



def return_item(index):

    for key, value in column_position.items():

        if value == index:

            return(key)
# is there a better way to return top values?

top_10 = pd.Series(np.matmul(W[35], H)).sort_values(ascending=False)[0:10]

top_10
[return_item(idx) for idx in top_10.index]
def prod_count(product_ids):



    prod_count = {}

    for item in product_ids:

        if item not in prod_count.keys():

            prod_count[item] = 1

        elif item in prod_count.keys():

            prod_count[item] += 1

    return prod_count
prod_count(product_list.product_id[35])
prod_matrix.shape
np.sum(prod_matrix.sum(axis=0)[0,:] > 3286)
import seaborn as sns

sns.distplot(prod_matrix.sum(axis=0)[0,:], bins = 30000, kde=False)

plt.xlim(0, 100)



plt.show()
prod_matrix.sum(axis=0)[:,580]
column_position[21386]
similar_users = product_list[product_list.user_id.isin(idx)]

similar_users
overlap = set(similar_users.product_id.iloc[0]) & set(similar_users.product_id.iloc[2])

overlap
counts = similar_users.product_id.apply(prod_count)
def id_values(row, overlap):

    for key, value in row.items():

        if key in overlap:

            print(key, value)
id_values(counts.iloc[0], overlap)
id_values(counts.iloc[2], overlap)
df = pd.concat([product_list.user_id, pd.DataFrame(W)], axis = 1)

df = pd.merge(test[0:10000], df, how = 'left').drop('eval_set', axis = 1)
df.iloc[1:10, 12]
x = df.iloc[:, 12:]

y = df.reordered.values

#del test, df
from statsmodels import api 

x = api.add_constant(x, prepend = False)

mn_log = api.MNLogit(y, x)

model = mn_log.fit()
model.summary()