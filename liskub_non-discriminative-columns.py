# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python



import pandas as pd

import numpy as np

import scipy.stats as stats



#Data

data = pd.read_csv('../input/train.csv')

#data.shape



#Two useful filters to discriminate between target values

filt0=data['target']==0

filt1=data['target']==1
# Let's create one column out of several columns ('ps_calc_15_bin', 'ps_calc_16_bin',

# 'ps_calc_17_bin', 'ps_calc_18_bin', 'ps_calc_19_bin', 'ps_calc_20_bin') 

# with no information loss.

ar=np.zeros(len(data))

u=int(1)

for k in ('ps_calc_15_bin', 'ps_calc_16_bin', 'ps_calc_17_bin', 'ps_calc_18_bin',

       'ps_calc_19_bin', 'ps_calc_20_bin') :

    for i in range(len(data)) :

        if data[k][i]==1 :

            ar[i]=ar[i]+u

    u=int(u*2)
# Calculate how frequentely the above values appear both for 'target'=1 and'target'=0

ct0=pd.crosstab(index=ar[filt0],columns='counts0')

ct1=pd.crosstab(index=ar[filt1],columns='counts1')
#Create contingency table

ob=pd.concat([ct0, ct1], axis=1)

obs=np.array([ob['counts0'],ob['counts1']])

obs=np.nan_to_num(obs)

#obs.astype(int)
#Run chi2 test

chi2, p, dof, expected = stats.chi2_contingency(obs)

print(p,'\n',chi2,'\n',dof,'\n')

#print(expected.astype(int))