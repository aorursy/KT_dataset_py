import pandas as pd
import numpy as np
import time as t
import math
cpu_start = t.clock()
real_start = t.time()

df = pd.read_csv('../input/train.csv', nrows=3500000)
dfp = pd.read_csv('../input/sample_submission.csv', nrows=3500000)
n = dfp.shape[0]

a=[] #actual demand
for line in df.itertuples():
    a.append(line.Demanda_uni_equil)
    
p=[] #predicted demans
for line in dfp.itertuples():
    p.append(line.Demanda_uni_equil)
    
x=0
for i in range(n):
    x += (math.pow((math.log1p(p[i] + 1) - math.log1p(a[i]+1)), 2))

e = math.sqrt((1/n)*(x))

print (e)
                    
cpu_time = t.clock()-cpu_start
real_time = t.time()-real_start
print('CPU: %s, Valós:%s' %(cpu_time, real_time))
