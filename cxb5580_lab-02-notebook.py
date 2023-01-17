# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

# Any results you write to the current directory are saved as output.
import numpy as np
"""
np.random.uniform(size=10)
np.random.uniform(3,4,5)
np.random.normal(size=4)
np.random.normal(-2,2,20)
np.random.binomial(10,0.1,size=2)
np.random.hypergeometric(10,20,5,10)
np.random.geometric(0.2,10)
np.random.negative_binomial(5,0.1,size=10)
np.random.poisson(5,10)
np.random.exponential(2,10)
np.random.chisquare(10,10)
np.random.f(3,4,20)
np.random.standard_t(3,10)
"""
from scipy.stats import binom
bnm=binom(100,0.1)
A=bnm.pmf([7,8,9])
A
sum(bnm.pmf(range(0,6)))
bnm.cdf(5)-bnm.cdf(0)
binom(100,0.01,1)

bnm.pmf([0,1,2,3,4])
from scipy.stats import poisson
poisson(1).pmf([0,1,2,3,4])
#Question 2 
from scipy.stats import norm
nrm1=norm(50,4)
sample1=nrm1.rvs(10000)
sample1

sample1.mean()
#verified that the mean is close to 50
from scipy.stats import nbinom
nbnm=nbinom(3,0.8)
sum(nbnm.pmf(range(5,11)))
import scipy.stats as stats #Question 3 Part 3
import pylab
nrm=norm(1000,5)
sample=nrm.rvs(2000)
stats.probplot(sample, dist="norm", plot=pylab)
pylab.show()
from scipy.stats import binom #Question 3 Part 1
bernoulli=binom(1000,0.5)
b=bernoulli.rvs(2000)
stats.probplot(b, dist="norm", plot=pylab)
pylab.show()
import scipy.stats as stats #Question 3 Part 2
import pylab
nrm=norm(1000,1)
sample=nrm.rvs(2000)
stats.probplot(sample, dist="norm", plot=pylab)
pylab.show()