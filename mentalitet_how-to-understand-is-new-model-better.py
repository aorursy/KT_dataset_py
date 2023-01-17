# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

from statsmodels.stats.weightstats import zconfint

from statsmodels.stats.weightstats import _zconfint_generic, _tconfint_generic

from math import sqrt

from scipy import stats

# Any results you write to the current directory are saved as output.
# CV for first model

kappabefore = np.array([0.8765, 0.8711, 0.8476, 0.8471, 0.9164]) #v12

scorebefore = np.array([4.4126, 4.0255, 5.0050, 4.4838, 3.9395]) #v12



# CV for second model

kappaafter = np.array([0.9028, 0.8792, 0.8715, 0.8756, 0.9123]) #v13

scoreafter = np.array([3.8182, 3.9054, 4.0769, 4.3103, 3.7365]) #v13
# Some fuctions for premutation test



def permutation_t_stat_1sample(sample, mean):

    t_stat = sum(list(map(lambda x: x - mean, sample)))

    return t_stat



def permutation_zero_distr_1sample(sample, mean, max_permutations = None):

    centered_sample = list(map(lambda x: x - mean, sample))

    if max_permutations:

        signs_array = set([tuple(x) for x in 2 * np.random.randint(2, size = (max_permutations, 

                                                                              len(sample))) - 1 ])

    else:

        signs_array =  itertools.product([-1, 1], repeat = len(sample))

    distr = [sum(centered_sample * np.array(signs)) for signs in signs_array] #####

    return distr



def permutation_test(sample, mean, max_permutations = None, alternative = 'two-sided'):

    if alternative not in ('two-sided', 'less', 'greater'):

        raise ValueError("alternative not recognized\n"

                         "should be 'two-sided', 'less' or 'greater'")

    

    t_stat = permutation_t_stat_1sample(sample, mean)

    

    zero_distr = permutation_zero_distr_1sample(sample, mean, max_permutations)

    

    if alternative == 'two-sided':

        return sum([1. if abs(x) >= abs(t_stat) else 0. for x in zero_distr]) / len(zero_distr)

    

    if alternative == 'less':

        return sum([1. if x <= t_stat else 0. for x in zero_distr]) / len(zero_distr)



    if alternative == 'greater':

        return sum([1. if x >= t_stat else 0. for x in zero_distr]) / len(zero_distr)
before_mean_std = kappabefore.std(ddof=1)/sqrt(len(kappabefore))

after_mean_std = kappaafter.std(ddof=1)/sqrt(len(kappaafter))

before_mean_std_score = scorebefore.std(ddof=1)/sqrt(len(scorebefore))

after_mean_std_score = scoreafter.std(ddof=1)/sqrt(len(scoreafter))

print('======================== KAPPA ========================')

print('mean kappa before {:.4f}'.format(kappabefore.mean()))

print('mean kappa after {:.4f}'.format(kappaafter.mean()))

print("model before mean kappa 95%% confidence interval", _tconfint_generic(kappabefore.mean(), before_mean_std,

                                                                       len(kappabefore) - 1,

                                                                       0.05, 'two-sided'))

print("model after mean kappa 95%% confidence interval", _tconfint_generic(kappaafter.mean(), after_mean_std,

                                                                       len(kappaafter) - 1,

                                                                       0.05, 'two-sided'))

print('======================== LOSS ========================')

print('mean score before {:.4f}'.format(scorebefore.mean()))

print('mean score after {:.4f}'.format(scoreafter.mean()))

print("model before mean loss 95%% confidence interval", _tconfint_generic(scorebefore.mean(), before_mean_std_score,

                                                                       len(scorebefore) - 1,

                                                                       0.05, 'two-sided'))

print("model after mean loss 95%% confidence interval", _tconfint_generic(scoreafter.mean(), after_mean_std_score,

                                                                       len(scoreafter) - 1,

                                                                       0.05, 'two-sided'))
print('======================== p-test KAPPA ========================')

_, p = stats.wilcoxon(kappabefore, kappaafter)

print('p-value WilcoxonResult test: %f' % p)

print("p-value permutation test: %f" % permutation_test(kappabefore - kappaafter, 0., max_permutations = 50000))



print('======================== p-test LOSS ========================')

_, p = stats.wilcoxon(scorebefore, scoreafter)

print('p-value WilcoxonResult test: %f' % p)

print("p-value permutation test: %f" % permutation_test(scorebefore - scoreafter, 0., max_permutations = 50000))