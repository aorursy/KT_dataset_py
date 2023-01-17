import numpy as np
import scipy.stats as st
import pandas as pd
data = pd.read_csv("../input/cryptocurrency-financial-data/consolidated_coin_data.csv")
data.head()
data = data['Market Cap']

# defining a function for remove ','(eg:1,00,000) and change format to int(100000)
def no_comma_yes_int(n):
    no_comma = n.replace(',','')
    yes_int = int(no_comma)
    return yes_int

# apply those function
data = data.apply(no_comma_yes_int)
#data = np.random.random(10000)
distributions =  [st.laplace, st.norm , st.expon]
# distributions = [        
#         st.alpha,st.anglit,st.arcsine,st.beta,st.betaprime,st.bradford,st.burr,st.cauchy,st.chi,st.chi2,st.cosine,
#         st.dgamma,st.dweibull,st.erlang,st.expon,st.exponnorm,st.exponweib,st.exponpow,st.f,st.fatiguelife,st.fisk,
#         st.foldcauchy,st.foldnorm,st.frechet_r,st.frechet_l,st.genlogistic,st.genpareto,st.gennorm,st.genexpon,
#         st.genextreme,st.gausshyper,st.gamma,st.gengamma,st.genhalflogistic,st.gilbrat,st.gompertz,st.gumbel_r,
#         st.gumbel_l,st.halfcauchy,st.halflogistic,st.halfnorm,st.halfgennorm,st.hypsecant,st.invgamma,st.invgauss,
#         st.invweibull,st.johnsonsb,st.johnsonsu,st.ksone,st.kstwobign,st.laplace,st.levy,st.levy_l,st.levy_stable,
#         st.logistic,st.loggamma,st.loglaplace,st.lognorm,st.lomax,st.maxwell,st.mielke,st.nakagami,st.ncx2,st.ncf,
#         st.nct,st.norm,st.pareto,st.pearson3,st.powerlaw,st.powerlognorm,st.powernorm,st.rdist,st.reciprocal,
#         st.rayleigh,st.rice,st.recipinvgauss,st.semicircular,st.t,st.triang,st.truncexpon,st.truncnorm,st.tukeylambda,
#         st.uniform,st.vonmises,st.vonmises_line,st.wald,st.weibull_min,st.weibull_max,st.wrapcauchy
#     ]
mles = []

for distribution in distributions:
    pars = distribution.fit(data)
    mle = distribution.nnlf(pars, data)
    mles.append(mle)

results = [(distribution.name, mle) for distribution, mle in zip(distributions, mles)]
best_fit = sorted(zip(distributions, mles), key=lambda d: d[1])[0]
print ('Best fit reached using {}, MLE value: {}'.format(best_fit[0].name, best_fit[1]))
#``````````````````````````````````````````````````````````````````````````````
# in this section we can discuss the mle

# now we put for only 3 variable x1, x2, x3 only
# now we can frame the equation

# L = (λ**n) * (E**(-(λ*(x1 + x2 + x3))))
# the above equation tells us the equation

from sympy import *

λ = Symbol('λ')
n = Symbol('n')
x1 = Symbol('x1')
x2 = Symbol('x2')
x3 = Symbol('x3')
expres = (λ**n) * (E**(-(λ*(x1 + x2 + x3))))

# simplify
print('MLE framed formula is',solve(diff(expres,λ),λ))

# calculate the value of λ
n = data.count()
sum_of_values = sum(data)
λ = (n/sum_of_values)

# in other form 
λ = format((n/sum_of_values),'.20f')

# print the equation in equation format

from sympy import *
n,x1,x2,x3 = symbols('n,x1,x2,x3')
init_printing()

n/(x1 + x2 + x3)