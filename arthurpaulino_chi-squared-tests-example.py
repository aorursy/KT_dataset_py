from scipy.stats.contingency import expected_freq

from scipy.stats import power_divergence

import pandas as pd

import numpy as np

import warnings



data = pd.read_csv('../input/heart.csv')

data.head()
lambdas = [

    'pearson', 'log-likelihood', 'freeman-tukey',

    'mod-log-likelihood', 'neyman', 'cressie-read'

]



def chi2(data, x, y):    

    observed = pd.crosstab(data[x], data[y])

    expected = pd.DataFrame(expected_freq(observed))

    expected.columns = observed.columns

    expected.index = observed.index

    

    if (observed < 5).sum().sum() > 0 or (expected < 5).sum().sum() > 0:

        # An often quoted guideline for the validity of this

        # calculation is that the test should be used only if

        # the observed and expected frequencies in each cell

        # are at least 5. (from SciPy docs)

        warnings.warn('Low count on observed or expected frequencies.')

    

    dof = expected.size - sum(expected.shape) + expected.ndim - 1

    delta_dof = observed.size - 1 - dof

    

    if dof == 0:

        tests = [{'lambda':'any', 'chi2':0, 'p':1}]

    else:

        if dof == 1:

            # Adjust `observed` according to Yates' correction for continuity.

            observed = observed + 0.5 * np.sign(expected - observed)

        tests = []

        for lambda_ in lambdas:

            chi2, p = power_divergence(

                observed, expected, ddof=delta_dof,

                axis=None, lambda_=lambda_)

            tests.append({'lambda':lambda_, 'chi2':chi2, 'p':p})

    tests = pd.DataFrame(tests)[['lambda', 'chi2', 'p']]

    return expected, observed, dof, tests
data['sex'].value_counts(ascending=True).plot(kind='bar')
expected, observed, dof, tests = chi2(data, 'sex', 'target')
expected.plot(kind='bar')
observed.plot(kind='bar')
tests