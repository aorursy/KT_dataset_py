import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

data = pd.read_csv('../input/pid-5M.csv')
data.head()

print( data.iat[0,0] , data.iat[0,1] , data.iat[0,2] )
print( data.iat[1,0] , data.iat[1,1] , data.iat[1,2] )
print( data.iat[2,0] , data.iat[2,1] , data.iat[2,2] )
print( len(data.index) )
h2 = plt.hist2d([1,2,3,4], [2,4,6,8], bins=[50,50])
bvp_all = data.plot.hexbin(x='p', y='beta', gridsize=40)
h1 = data.hist('p', bins=100)
m2Min = -0.1
m2Max = 1.4
nM2Bins = 200
m2BinCounts = [0]*nM2Bins

for ev in range(0, int(0.001*len(data.index))):
    pid = data.iat[ev, 0]
    p = data.iat[ev, 1]
    b = data.iat[ev, 3]
    m2 = p*p*(1 - b*b)/(b*b)
    m2Bin = int(math.floor( (m2 - m2Min)/((m2Max - m2Min)/nM2Bins) ))
    if m2Bin >= 0 and m2Bin < nM2Bins:
        m2BinCounts[m2Bin] += 1
    
#hm2 = plt.bar(np.linspace(m2Min, m2Max, nM2Bins), m2BinCounts)
hm2 = plt.bar(range(0, nM2Bins), m2BinCounts)
#list(range(0, 5))
#np.linspace(m2Min, m2Max, nM2Bins)