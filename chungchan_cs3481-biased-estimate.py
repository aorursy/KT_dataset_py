from scipy import stats

import numpy as np
m = 2

#p_list = stats.uniform.rvs(size=m)

p_list = np.array([0.5,0.5])

#p_list = np.ones(m) * 0.5
N = 5000

n_list = np.arange(1,N+1)
k = 20

phat_list = np.array([[stats.binom.rvs(n,p,size=k)/n for n in n_list] for p in p_list])

phat_list.shape
%matplotlib inline

import matplotlib.pyplot as plt

# set figure size

plt.rcParams["figure.figsize"] = (15,12)

# produce vector inline graphics

from IPython.display import set_matplotlib_formats

set_matplotlib_formats('pdf', 'png')
! sudo apt-get install texlive-latex-recommended 

! sudo apt install texlive-latex-extra

! sudo apt install dvipng

plt.rcParams['text.usetex'] = True
# plot the maximum of the probability estimates of each coin

for i in range(phat_list.shape[2]):

  plt.plot(n_list,phat_list.max(axis=0)[:,i],

          linestyle='',marker='.',color='blue',markersize=1)

# plot the maximum of true probabilities of head

plt.axhline(p_list.max(),color='red',label=r'$\max_i p_i$')

plt.ylim([0,1])

plt.xlim([0,len(n_list)])

plt.title(r'Plot of $\max_i\hat{p}_i$ vs sample size')

plt.xlabel('sample size')

plt.ylabel('probability')

plt.legend()

plt.show()