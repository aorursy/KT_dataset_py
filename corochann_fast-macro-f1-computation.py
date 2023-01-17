import numba as nb

import numpy as np

import pandas as pd

from sklearn import metrics



from contextlib import contextmanager

from time import perf_counter





@contextmanager

def timer(name):

    t0 = perf_counter()

    yield

    t1 = perf_counter()

    print('[{}] done in {:.3f} s'.format(name, t1-t0))

def macro_f1_score(y_true, y_pred, n_labels):

    total_f1 = 0.

    for i in range(n_labels):

        yt = y_true == i

        yp = y_pred == i



        tp = np.sum(yt & yp)



        tpfp = np.sum(yp)

        tpfn = np.sum(yt)

        if tpfp == 0:

            print('[WARNING] F-score is ill-defined and being set to 0.0 in labels with no predicted samples.')

            precision = 0.

        else:

            precision = tp / tpfp

        if tpfn == 0:

            print(f'[ERROR] label not found in y_true...')

            recall = 0.

        else:

            recall = tp / tpfn



        if precision == 0. or recall == 0.:

            f1 = 0.

        else:

            f1 = 2 * precision * recall / (precision + recall)

        total_f1 += f1

    return total_f1 / n_labels
macro_f1_score_nb = nb.jit(nb.float64(nb.int32[:], nb.int32[:], nb.int64), nopython=True, nogil=True)(macro_f1_score)
n_class = 10

datasize = 5_000_000

y_true = np.random.randint(0, n_class, datasize).astype(np.int32)

y_pred = np.random.randint(0, n_class, datasize).astype(np.int32)
with timer('sklearn'):

    f1_sk = metrics.f1_score(y_true, y_pred, average='macro')

with timer('custom'):

    f1_custom = macro_f1_score(y_true, y_pred, n_class)

with timer('custom numba'):

    f1_custom_nb = macro_f1_score_nb(y_true, y_pred, n_class)
print('f1_sk', f1_sk, 'f1_custom', f1_custom, 'f1_custom_nb', f1_custom_nb)