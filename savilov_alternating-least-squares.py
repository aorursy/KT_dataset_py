import pandas as pd

import implicit

import os

import numpy as np

from scipy import sparse
purchases_train = pd.read_csv('/kaggle/input/purchases_train.csv')

customers = pd.read_csv('/kaggle/input/customers.csv')
purchases_train = purchases_train.merge((customers), left_on='customer_id', right_on='customer_id', how='left')
purchases_train = purchases_train.fillna(purchases_train['sex'].value_counts().index[0])
purchases_train['sex'] = purchases_train['sex'].replace(['Female','Male'], [0,1])
purchases_train
user_items = sparse.coo_matrix(

    (

        purchases_train.sex,

        (

            purchases_train.customer_id,

            purchases_train.product_id

        )

    )

).tocsr()
item_users = user_items.T.tocsr()
model = implicit.als.AlternatingLeastSquares(factors=64, iterations=100)
np.random.seed(42)

model.fit(item_users=item_users)
purchases_test = pd.read_csv('/kaggle/input/purchases_test.csv')

display(

    purchases_test.head(),

)
relevant = purchases_test.groupby('customer_id')['product_id'].apply(lambda s: s.values).reset_index()

relevant.rename(columns={'product_id': 'product_ids'}, inplace=True)

relevant.head()
recommendations = []

for user_id in relevant['customer_id']:

    recommendations.append([x[0] for x in model.recommend(userid=user_id, user_items=user_items, N=10)])
def apk(actual, predicted, k=10):

    """

    Computes the average precision at k.

    This function computes the average prescision at k between two lists of

    items.

    Parameters

    ----------

    actual : list

             A list of elements that are to be predicted (order doesn't matter)

    predicted : list

                A list of predicted elements (order does matter)

    k : int, optional

        The maximum number of predicted elements

    Returns

    -------

    score : double

            The average precision at k over the input lists

    """

    if len(predicted)>k:

        predicted = predicted[:k]



    score = 0.0

    num_hits = 0.0



    for i,p in enumerate(predicted):

        if p in actual and p not in predicted[:i]:

            num_hits += 1.0

            score += num_hits / (i+1.0)



    if len(actual) == 0:

        return 0.0



    return score / min(len(actual), k)



def mapk(actual, predicted, k=10):

    """

    Computes the mean average precision at k.

    This function computes the mean average prescision at k between two lists

    of lists of items.

    Parameters

    ----------

    actual : list

             A list of lists of elements that are to be predicted 

             (order doesn't matter in the lists)

    predicted : list

                A list of lists of predicted elements

                (order matters in the lists)

    k : int, optional

        The maximum number of predicted elements

    Returns

    -------

    score : double

            The mean average precision at k over the input lists

    """

    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])
mapk(relevant['product_ids'], recommendations, k=10)