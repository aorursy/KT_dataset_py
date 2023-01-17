!apt-get install -y libopenblas-dev pbzip2

!pip install -U tabulate==0.8.2

!pip install h2o4gpu

import h2o4gpu
import gc

import numpy as np

import scipy

from sklearn.model_selection import train_test_split



files = [

    '../input/combined_data_1.txt',

    '../input/combined_data_2.txt',

    '../input/combined_data_3.txt',

    '../input/combined_data_4.txt',

]



coo_row = []

coo_col = []

coo_val = []



for file_name in files:

    print('processing {0}'.format(file_name))

    with open(file_name, "r") as f:

        movie = -1

        for line in f:

            if line.endswith(':\n'):

                movie = int(line[:-2]) - 1

                continue

            assert movie >= 0

            splitted = line.split(',')

            user = int(splitted[0])

            rating = float(splitted[1])

            coo_row.append(user)

            coo_col.append(movie)

            coo_val.append(rating)

    gc.collect()



print('transformation...')



coo_val = np.array(coo_val, dtype=np.float32)

coo_col = np.array(coo_col, dtype=np.int32)

coo_row = np.array(coo_row)

user, indices = np.unique(coo_row, return_inverse=True)

user = user.astype(np.int32)



gc.collect()



coo_matrix = scipy.sparse.coo_matrix((coo_val, (indices, coo_col)))

shape = coo_matrix.shape

print('R matrix size', shape)



gc.collect()



print('splitting into training and validation set')

train_row, test_row, train_col, test_col, train_data, test_data = train_test_split(

    coo_matrix.row, coo_matrix.col, coo_matrix.data, test_size=0.2, random_state=42)



train = scipy.sparse.coo_matrix(

    (train_data, (train_row, train_col)), shape=shape)

test = scipy.sparse.coo_matrix(

    (test_data, (test_row, test_col)), shape=shape)

n_components = 40

_lambda = 0.01

# increase it in case out-of GPU memory, but n_components / BATCHES has to be a multiple of 10

BATCHES=1







scores = []

factorization = h2o4gpu.solvers.FactorizationH2O(

    n_components, _lambda, max_iter=100)

factorization.fit(train, X_test=test, X_BATCHES=BATCHES,

                      THETA_BATCHES=BATCHES, scores=scores, verbose=True, early_stopping_rounds=5)

print('best iteration:',factorization.best_iteration)
print('X shape:', factorization.XT.shape)

print('ThetaT shape:', factorization.thetaT.shape)
