!pip install dit
import numpy as np

import dit
p = np.random.random() # randomly generate the probability of head coming up.

coin_flip = dit.Distribution(['H','T'],[p,1-p]) # create the probability model 

coin_flip
coin_flip.rand(), coin_flip.rand(10)
N=5000
%matplotlib inline

import matplotlib.pyplot as plt

# set figure size

plt.rcParams["figure.figsize"] = (8,4)

# produce vector inline graphics

from IPython.display import set_matplotlib_formats

set_matplotlib_formats('pdf', 'svg')
n_list = np.arange(1,N+1) 

# error boundaries defined by 2 standard deviations away from mean 

sigma2_list = np.sqrt(p*(1-p)/n_list) * 2 # numpy list support elementwise operations

ub = p+sigma2_list # upper boundary

lb = p-sigma2_list # lower boundary



f, ax = plt.subplots()

ax.axhline(p,color='red')

ax.fill_between(n_list, lb, ub, color='red',alpha=0.2,label=r'$p\pm 2\sigma$')

ax.set_ylim([0,1])

ax.set_xlim([0,len(n_list)])

ax.set_title('Plot of empirical probability vs sample size')

ax.set_xlabel('sample size')

ax.set_ylabel('probability')

ax.legend()
outcome_list = coin_flip.rand(N)
phat_list = np.array(

    [(0,1)[outcome == 'H'] for outcome in outcome_list],  # see tenary operator 

                  # https://book.pythontips.com/en/latest/ternary_operators.html

    dtype=float).cumsum()/n_list # numpy number list supports cumulative sum
outcome_list[:5], phat_list[:5] # for sanity check
ax.plot(n_list,phat_list,marker='.',color='blue',linestyle='',markersize=1) 

f