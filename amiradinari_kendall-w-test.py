# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import math
import sys
from matplotlib.path import Path
import matplotlib.patches as patches
import scipy.linalg as lin
import matplotlib.pyplot as plt
from sklearn import mixture
import pandas as pd
import scipy
import spm1d
import warnings
from sklearn.covariance import LedoitWolf
from scipy import stats

import warnings
from collections import namedtuple
from numpy import array, asarray, ma
from scipy._lib.six import callable, string_types
from scipy._lib._version import NumpyVersion
from scipy._lib._util import _lazywhere
import scipy.special as special
def kendalltau_costum(x, Y, initial_lexsort=None, nan_policy='propagate', method='auto'):
    """
    Calculate Kendall's tau of two data sets, It tests wether data set x has the same correlation matrix as data set Y .
    Parameters
    ----------
    x, y : array_like
        Arrays of rankings, of the same shape. If arrays are not 1-D, they will
        be flattened to 1-D.
    initial_lexsort : bool, optional
        Unused (deprecated).
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains nan. 'propagate' returns nan,
        'raise' throws an error, 'omit' performs the calculations ignoring nan
        values. Default is 'propagate'. Note that if the input contains nan
        'omit' delegates to mstats_basic.kendalltau(), which has a different
        implementation.
    method: {'auto', 'asymptotic', 'exact'}, optional
        Defines which method is used to calculate the p-value [5]_.
        'asymptotic' uses a normal approximation valid for large samples.
        'exact' computes the exact p-value, but can only be used if no ties
        are present. 'auto' is the default and selects the appropriate
        method based on a trade-off between speed and accuracy.
    Returns
    -------
   
    pvalue : float
       The two-sided p-value for a hypothesis test whose null hypothesis is
       an absence of association, tau = 0.
   
    """
    x = np.asarray(x)
    yi=[]
    Y=np.asarray(Y)
    if (x.shape[1]>x.shape[0]):
        cov_0 =np.asarray(LedoitWolf().fit(Y).covariance_)
        tri_lower_diag = np.linalg.cholesky(cov_0)
        for i in range(0,x.shape[0]):
          yi.append(np.matmul(lin.inv(tri_lower_diag),x[i]))
        yi=np.asarray(yi)
        S_y=LedoitWolf().fit(yi).covariance_
        I=np.identity(x.shape[1])
        W=((1/x.shape[1])*(np.matmul(S_y-I,S_y-I)).trace())-((x.shape[1]/x.shape[0])*(S_y.trace()/x.shape[1])*(S_y.trace()/x.shape[1]))+(x.shape[1]/x.shape[0])
        p_value = 1 - stats.chi2.cdf(x=((((x.shape[0]*W-x.shape[1])*x.shape[1])/2)+x.shape[1]),  # Find the p-value
                             df=(x.shape[1]*(x.shape[1]+1))/2)
    else :
         cov_0 =np.asarray(np.cov(Y.T))
         tri_lower_diag = np.linalg.cholesky(cov_0)
         for i in range(0,x.shape[0]):
           yi.append(np.matmul(lin.inv(tri_lower_diag),x[i]))
         yi=np.asarray(yi)
         S_y=np.cov(yi.T)
         I=np.identity(x.shape[1])
         W=((1/x.shape[1])*(np.matmul(S_y-I,S_y-I)).trace())-((x.shape[1]/x.shape[0])*(S_y.trace()/x.shape[1])*(S_y.trace()/x.shape[1]))+(x.shape[1]/x.shape[0])
         p_value = 1 - stats.chi2.cdf(x=((((x.shape[0]*W-x.shape[1])*x.shape[1])/2)+x.shape[1]),  # Find the p-value
                             df=(x.shape[1]*(x.shape[1]+1))/2)
        
    return(p_value)
# Any results you write to the current directory are saved as output.
