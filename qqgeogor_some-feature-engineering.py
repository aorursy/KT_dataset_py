import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy import sparse as ssp

from scipy.stats import spearmanr

from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler,MinMaxScaler

from sklearn.decomposition import PCA, FastICA,TruncatedSVD,NMF

from sklearn.random_projection import GaussianRandomProjection

from sklearn.random_projection import SparseRandomProjection

from sklearn.neural_network import MLPRegressor

from sklearn.pipeline import make_pipeline

from sklearn.manifold import TSNE

from sklearn.cluster import KMeans

from sklearn.model_selection import KFold,StratifiedKFold

path = '../input/'



seed =1024

np.random.seed(1024)





smooth=5

# read datasets

train = pd.read_csv(path+'train.csv')

test = pd.read_csv(path+'test.csv')



# process columns, apply LabelEncoder to categorical features



categorical = []

for c in train.columns:

    if train[c].dtype == 'object':

        lbl = LabelEncoder() 

        lbl.fit(list(train[c].values) + list(test[c].values)) 

        train[c] = lbl.transform(list(train[c].values))

        test[c] = lbl.transform(list(test[c].values))

        categorical.append(c)









y = train["y"]

y = np.log(y+smooth)

y_mean = np.mean(y)

y = y-y_mean

train["y"] = y







X = train.drop('y', axis=1).copy(deep=True)

X_t = test.copy(deep=True)

X = X.drop(categorical,axis=1)

X_t = X_t.drop(categorical,axis=1)



scaler = MinMaxScaler()

X['ID'] = scaler.fit_transform(X[['ID']].values)

X_t['ID'] = scaler.transform(X_t[['ID']].values)



print('feature 1')

sum_cols = []

for c in X.columns:

    score = (spearmanr(y,X[c]))

    if score[0]>=0.2 and score[0]<=0.3:

        print(c,score)

        sum_cols.append(c)



X['sum_row_2_to_3'] = X.drop('ID', axis=1)[sum_cols].sum(axis=1)

X_t['sum_row_2_to_3'] = X_t.drop('ID', axis=1)[sum_cols].sum(axis=1)

print('new feature',spearmanr(y,X['sum_row_2_to_3']))



print('feature 2')

sum_cols = []

for c in X.columns:

    score = (spearmanr(y,X[c]))

    if score[0]>=0.1 and score[0]<=0.2:

        print(c,score)

        sum_cols.append(c)



X['sum_row_1_to_2'] = X.drop('ID', axis=1)[sum_cols].sum(axis=1)

X_t['sum_row_1_to_2'] = X_t.drop('ID', axis=1)[sum_cols].sum(axis=1)

print('new feature',spearmanr(y,X['sum_row_1_to_2']))



print('feature 3')

sum_cols = []

for c in X.columns:

    score = (spearmanr(y,X[c]))

    if score[0]>=0.05 and score[0]<=0.1:

        print(c,score)

        sum_cols.append(c)



X['sum_row_05_to_1'] = X.drop('ID', axis=1)[sum_cols].sum(axis=1)

X_t['sum_row_05_to_1'] = X_t.drop('ID', axis=1)[sum_cols].sum(axis=1)

print('new feature',spearmanr(y,X['sum_row_05_to_1']))