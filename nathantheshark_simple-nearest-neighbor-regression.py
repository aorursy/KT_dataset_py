import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



%matplotlib inline
train = pd.read_csv('../input/train_2016.csv', parse_dates=['transactiondate'])

train.shape
prop = pd.read_csv('../input/properties_2016.csv')

prop.shape
prop.head()
prop.columns
X = prop[['parcelid', 'bathroomcnt', 'bedroomcnt', 'calculatedfinishedsquarefeet']]

X.head()

X = X.dropna()

X = X[(X.bathroomcnt > 0) & (X.bedroomcnt > 0) & (X.calculatedfinishedsquarefeet > 0)]

X.shape
X.head()
X = pd.merge(X, train, on='parcelid')

X.shape
X.sort_values(by=['parcelid', 'transactiondate'], inplace=True)

X.head()
#X['transaction_year'] = X['transactiondate'].dt.year

#X['transaction_month'] = X['transactiondate'].dt.month

X['transaction_yearmonth'] = 100 * X['transactiondate'].dt.year + X['transactiondate'].dt.month

X.head()
min_transaction_yearmonth = X.transaction_yearmonth.min()

X['transaction_yearmonth_i'] = X.transaction_yearmonth - min_transaction_yearmonth

np.sort(X.transaction_yearmonth_i.unique())
y = X.logerror

X = X[['bathroomcnt', 'bedroomcnt', 'calculatedfinishedsquarefeet', 'transaction_yearmonth_i']]

print ("X shape =", X.shape)

print ("y shape =", y.shape)

X.head()
from sklearn import neighbors

from sklearn.model_selection import cross_val_score



n_neighbors_lst = np.arange(1, 100+1, 1)

mae_lst = []

std_lst = []



for n_neighbors in n_neighbors_lst: 

    #n_neighbors = 5

    weights = 'distance'

    knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)

    scores = cross_val_score(knn, X, y, scoring='neg_mean_absolute_error', cv=5)

    scores = np.fabs(scores)

    mean_score = np.mean(scores)

    std_score = np.std(scores)

    

    mae_lst.append(mean_score)

    std_lst.append(std_score)

#knn.fit(X, y)

#y_hat = knn.predict(X)



#mae = np.mean(np.fabs(y - y_hat))

#mae_lst.append(mae)



plt.plot(n_neighbors_lst, mae_lst, linewidth=2)

plt.title('MAE by number of neighbors')

plt.xlabel('k')

plt.ylabel('MAE')



plt2 = plt.twinx()

plt2.plot(n_neighbors_lst, std_lst, linewidth=2, color='red')