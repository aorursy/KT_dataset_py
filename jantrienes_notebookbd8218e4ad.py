%matplotlib inline

import pandas as pd
s = pd.Series(data=[5850000, 6000000, 5700000, 13100000, 16331452], name='price_doc')

s
def _revrt(X,m=None):

    """

    Inverse of forrt. Equivalent to Munro (1976) REVRT routine.

    """

    if m is None:

        m = len(X)

    i = int(m // 2+1)

    y = X[:i] + np.r_[0,X[i:],0]*1j

    return np.fft.irfft(y)*m



from statsmodels.nonparametric import kdetools

import numpy as np



kdetools.revrt = _revrt
import statsmodels

import seaborn as sns

print(statsmodels.__version__)

print(sns.__version__)
_ = sns.distplot(s, bins=2, kde=True)