import subprocess

import re

import sys

import os

import glob

import warnings

import ctypes



_MKL_ = 'mkl'

_OPENBLAS_ = 'openblas'





class BLAS:

    def __init__(self, cdll, kind):

        if kind not in (_MKL_, _OPENBLAS_):

            raise ValueError(f'kind must be {MKL} or {OPENBLAS}, got {kind} instead.')

        

        self.kind = kind

        self.cdll = cdll

        

        if kind == _MKL_:

            self.get_n_threads = cdll.MKL_Get_Max_Threads

            self.set_n_threads = cdll.MKL_Set_Num_Threads

        else:

            self.get_n_threads = cdll.openblas_get_num_threads

            self.set_n_threads = cdll.openblas_set_num_threads

            



def get_blas(numpy_module):

    LDD = 'ldd'

    LDD_PATTERN = r'^\t(?P<lib>.*{}.*) => (?P<path>.*) \(0x.*$'



    NUMPY_PATH = os.path.join(numpy_module.__path__[0], 'core')

    MULTIARRAY_PATH = glob.glob(os.path.join(NUMPY_PATH, '_multiarray_umath.*so'))[0]

    ldd_result = subprocess.run(

        args=[LDD, MULTIARRAY_PATH], 

        check=True,

        stdout=subprocess.PIPE, 

        universal_newlines=True

    )



    output = ldd_result.stdout



    if _MKL_ in output:

        kind = _MKL_

    elif _OPENBLAS_ in output:

        kind = _OPENBLAS_

    else:

        return



    pattern = LDD_PATTERN.format(kind)

    match = re.search(pattern, output, flags=re.MULTILINE)



    if match:

        lib = ctypes.CDLL(match.groupdict()['path'])

        return BLAS(lib, kind)

    



class single_threaded:

    def __init__(self, numpy_module=None):

        if numpy_module is not None:

            self.blas = get_blas(numpy_module)

        else:

            import numpy

            self.blas = get_blas(numpy)



    def __enter__(self):

        if self.blas is not None:

            self.old_n_threads = self.blas.get_n_threads()

            self.blas.set_n_threads(1)

        else:

            warnings.warn(

                'No MKL/OpenBLAS found, assuming NumPy is single-threaded.'

            )



    def __exit__(self, *args):

        if self.blas is not None:

            self.blas.set_n_threads(self.old_n_threads)

            if self.blas.get_n_threads() != self.old_n_threads:

                message = (

                    f'Failed to reset {self.blas.kind} '

                    f'to {self.old_n_threads} threads (previous value).'

                )

                raise RuntimeError(message)

    

    def __call__(self, func):

        def _func(*args, **kwargs):

            self.__enter__()

            func_result = func(*args, **kwargs)

            self.__exit__()

            return func_result

        return _func
import numpy

from tqdm.auto import tqdm

from sklearn.model_selection import cross_val_score

from sklearn.datasets import make_classification

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
X, y = make_classification(

    n_samples=700,

    n_features=40,

    n_informative=40,

    n_redundant=0,

    n_repeated=0,

    n_classes=2,

    n_clusters_per_class=2,

    flip_y=0.05,

    class_sep=2.0,

    hypercube=True,

    random_state=577

)
# @single_threaded(numpy)

def cv_qda(X, y):

    return cross_val_score(

        QuadraticDiscriminantAnalysis(),

        X, y,

        cv=32,

        scoring="roc_auc",

        n_jobs=1,

        verbose=0

    ).mean()
%%time

for _ in tqdm(range(512 * 4)):

    _ = cv_qda(X, y)
%%time

with single_threaded(numpy):

    for _ in tqdm(range(512 * 4)):

        _ = cv_qda(X, y)