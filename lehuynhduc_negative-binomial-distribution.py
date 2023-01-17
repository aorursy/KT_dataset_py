import numpy as np

r = 2 

p = 0.3

S = np.random.negative_binomial(r,p,10000)

m = S.mean() # = r_*(1-p_)/p_

v = S.var() # = r_*(1-p_)/p_**2

p_= m/v

r_ = m**2/(v-m)

print(r_,p_)