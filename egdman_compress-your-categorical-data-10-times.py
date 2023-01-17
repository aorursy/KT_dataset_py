import numpy as np

import pandas as pd

from itertools import chain

import scipy.sparse as spar





def sparse_dummies(categories):

    num_rows = categories.shape[0]

    num_cats = len(categories.values.categories)

    if num_cats == 0: return None

    categories = categories.reset_index(drop=True)[categories.values.codes > -1]

    data = np.ones(categories.shape[0])

    return spar.csc_matrix(

        (data, (categories.index.values, categories.values.codes)),

        shape=(num_rows, num_cats))





def one_hot(df):

    df = df.apply(lambda col: col.astype('category'), axis=0)

    cat_counts = df.apply(lambda col: len(col.values.categories), axis=0)

    header = list(chain.from_iterable(

    [[colname] * ncats for colname, ncats in zip(cat_counts.index, cat_counts.values)]))



    dummy_matrices = (sparse_dummies(df[col]) for col in df.columns)

    # drop None's

    dummy_matrices = list(elem for elem in dummy_matrices if elem is not None)

    

    if len(dummy_matrices) == 0: return None, None

    mtx = spar.hstack(dummy_matrices)

    

    return header, mtx





def save_ohe(filename, header, matrix):

    np.savez(filename,

        header=header,

        indices=matrix.indices,

        indptr=matrix.indptr,

        shape=matrix.shape)





def load_ohe(filename):

    loader = np.load(filename)

    header = loader['header']

    indices = loader['indices']

    indptr = loader['indptr']

    data = np.ones(len(indices))

    mtx = spar.csc_matrix((data, indices, indptr), shape = loader['shape'])

    return header, mtx





def chunker(seq, size):

    return (seq[pos:pos + size] for pos in range(0, len(seq), size))
cat_path = '../input/train_categorical.csv'

# cat_path = '../input/test_categorical.csv'



NROWS = 50000 # test on 1st 50 000 rows (the entire dataset will probably take too long for kaggle)



chunksize = 30 # number of columns to read at once



columns = pd.read_csv(

    cat_path,

    index_col=0,

    nrows = 1,

    dtype=str).columns.values



ncols = len(columns)

print(ncols)



headers = []

matrices = []

progress = 0

# read file in vertical chunks

for col_subset in chunker(columns, chunksize):

    progress += chunksize

    

    usecols = np.append(['Id'], col_subset)

    df = pd.read_csv(

        cat_path,

        index_col=0,

        dtype=str,

        nrows=NROWS,

        usecols=usecols

    )

    

    header, mtx = one_hot(df)

    if mtx is not None:

        matrices.append(mtx)

        headers.append(header)

        print("progress = {:.1f}% : {}".format(100.*progress / (1.*ncols), mtx.shape))





full_header = np.hstack(headers)

full_mtx = spar.hstack(matrices).tocsc()



print(full_header.shape)

print(full_mtx.shape)



with open('train_cat_sparse', 'wb') as resfile:

    save_ohe(resfile, full_header, full_mtx)
header, mtx = load_ohe('train_cat_sparse')



print(header.shape)

print(mtx.shape)



# use this matrix with xgboost

from xgboost import XGBClassifier

from sklearn.model_selection import cross_val_score, StratifiedKFold

from sklearn.metrics import matthews_corrcoef, roc_auc_score



num_path = '../input/train_numeric.csv'

labels = pd.read_csv(num_path, usecols=[969], nrows = NROWS, dtype=np.float32)



X = mtx

y = labels.values.ravel()

y_pred = np.empty(y.shape)



clf = XGBClassifier()



kfold = StratifiedKFold(n_splits=4, shuffle=True)

for (train, test) in kfold.split(X, y):

    y_pred[test] = clf.fit(X[train], y[train]).predict_proba(X[test])[:,1]

    print("ROC AUC = {:.3f}".format(roc_auc_score(y[test], y_pred[test])))

    

print()     

print("FINAL SCORE = {:.3f}".format(roc_auc_score(y, y_pred)))