! pip install --upgrade pip

! pip install hvplot
from IPython.display import Image

Image('/kaggle/input/segmentation-examples/ABCDE_Facet.jpg')
Image('/kaggle/input/segmentation-examples/FGE_Facet.jpg')
import pandas as pd

import numpy as np 

from functools import partial

from toolz.curried import *

from sklearn.metrics import silhouette_score, pairwise_distances

from sklearn.preprocessing import StandardScaler

from scipy.optimize import fminbound

from typing import Tuple

from itertools import chain

import hvplot.pandas





def yield_rule(X: np.ndarray, D: np.ndarray) -> Tuple[float, float]:

    not_isnan_mask = np.any(~np.isnan(X), axis=-1)

    criterion = partial(silhouette_score, X = D[not_isnan_mask, :][:, not_isnan_mask], metric = 'precomputed', njobs=-1)



    xopt_ = []

    fval_ = []



    for c in range(X.shape[1]):

        xopt, fval, _, _= fminbound(func = lambda x: -criterion(labels = X[not_isnan_mask,c] > x), 

                                    x1=np.nanquantile(X[not_isnan_mask,c], 0.1), x2=np.nanquantile(X[:,c], 0.9), 

                                    full_output=True)



        xopt_.append(xopt)

        fval_.append(-fval)



    split_feature = np.argmax(fval_)

    score = fval_[split_feature]

    split_point= xopt_[split_feature]

    

    return  split_feature, split_point



def yeild_split(X: np.ndarray, split_feature: int, split_point: float):

        

    left_ = np.where(X[:, [split_feature]] > split_point, X, np.full_like(X, np.nan, dtype=np.float))

    right_ = np.where(X[:, [split_feature]] <= split_point, X, np.full_like(X, np.nan, dtype=np.float))

    

    return left_, right_



def rule_split_chain(x, D):

    split_feature, split_point = yield_rule(x, D)

    return yeild_split(x, split_feature, split_point)
data = pd.read_csv('/kaggle/input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')

data
depth = 3



X = StandardScaler().fit_transform(data.loc[:,['Age','Annual Income (k$)','Spending Score (1-100)']].to_numpy())

D = pairwise_distances(X, n_jobs=-1)

splitter = compose_left(lambda t: chain(*t), 

                        map(partial(rule_split_chain, D=D)))

leaves = pipe(reduce(lambda x, _: splitter(x), [[[X]] for _ in range(depth)]), lambda x: chain(*x), list)

labels = np.stack((np.any(~np.isnan(branch), axis=-1) for branch in leaves)).T.astype(float).argmax(-1)

silhouette_score(labels=labels, X = D, metric = 'precomputed', njobs=-1)
((data.loc[:,['Annual Income (k$)','Spending Score (1-100)']]

 .assign(label = labels.astype(str))

 .hvplot.scatter(x='Annual Income (k$)',y='Spending Score (1-100)', color='label'))



+ 



(data.loc[:,['Age','Spending Score (1-100)']]

 .assign(label = labels.astype(str))

 .hvplot.scatter(x='Age',y='Spending Score (1-100)', color='label'))



+ 



(data.loc[:,['Annual Income (k$)','Age']]

 .assign(label = labels.astype(str))

 .hvplot.scatter(x='Annual Income (k$)',y='Age', color='label'))



+



(data.loc[:,['Gender','Age']]

 .assign(label = labels.astype(str))

 .hvplot.scatter(x='Gender',y='Age', color='label'))).cols(2).opts(title='Tree segmentation')