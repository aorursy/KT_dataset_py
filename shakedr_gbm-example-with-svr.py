# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv("../input/train.csv")
X, y = df_train[['PoolArea', 'LotArea']].values, df_train['SalePrice'].values

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.svm import SVR
gbm = GradientBoostingRegressor(max_depth=3).fit(X, y)

svr = SVR().fit(X, y)
((svr.predict(X) - y)**2).mean()**0.5
leafs = gbm.apply(X)



trees_svrs = []



for tree_num in range(leafs.shape[1]):

    svrs = {}

    

    for leaf_idx in np.unique(leafs[:, tree_num]):

        flt = (leafs[:, tree_num] == leaf_idx)

        

        svrs[leaf_idx] = SVR().fit(X[flt], y[flt])

        

    trees_svrs.append(svrs)
trees_svrs[24][9]
def predict(X):

    leafs = gbm.apply(X)

    y = np.zeros(X.shape[0])



    for tree_num in range(leafs.shape[1]):

        

        for i in range(leafs.shape[0]):

            

            svr = trees_svrs[tree_num][leafs[i, tree_num]]

            y[i] += svr.predict(X[i:i+1])

            

    return y
predict(X)