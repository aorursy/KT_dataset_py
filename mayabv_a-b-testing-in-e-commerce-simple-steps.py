import numpy as np

from scipy import stats

from scipy.stats import norm

from scipy.stats import chi2





n_A = 5000

n_B = 3000



mu_A = 65.

mu_B = 68.



std_A = 42.

std_B = 45.





Z = (mu_B - mu_A)/np.sqrt(std_B**2/n_B + std_A**2/n_A)

pvalue = 1-norm.cdf(Z)



print("z-statistic: {0}\np-value: {1}".format(Z,pvalue))
from IPython.display import Image

Image("/kaggle/input/pvalues.jpg")
#The expected outcomes and the remaining metrics can be calculated in Python:



observed = [[145.,127],[4855.,2873]]

stats.chi2_contingency(observed)
