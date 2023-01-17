# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
datam=pd.read_csv(os.path.join(dirname, filename))

datam
#Since degree_t is of categorical type, we have to make it numeral type

#So as to perform the computation

#for this we are making an entire different datatype

datam1=datam.copy(deep=True)
dummy=pd.get_dummies(datam1["degree_t"])

dummy
datam1=pd.concat([datam1,dummy], axis=1)

datam1
datat=pd.concat([datam1['mba_p'],dummy], axis=1)

datat
import scipy.stats as stats

# stats f_oneway functions takes the groups as input and returns F and P-value

fvalue, pvalue = stats.f_oneway(datat['Comm&Mgmt'], datat['Others'], datat['Sci&Tech'], datat['mba_p'])

print(fvalue, pvalue)
data_new=pd.melt(datat.reset_index(), id_vars=['index'], value_vars=['mba_p',"Sci&Tech",'Others','Comm&Mgmt'])

data_new.columns=['index','treatments','value']

data_new
import statsmodels.api as sm

from statsmodels.formula.api import ols
model=ols('value ~ C(treatments)', data=data_new).fit()
model.summary()
anov_table=sm.stats.anova_lm(model, typ=1)

anov_table
#to know the pairs of signufucant different treaments

#lets perform mmultiple pairwise comparison

from statsmodels.stats.multicomp import pairwise_tukeyhsd



m_comp=pairwise_tukeyhsd(endog=data_new['value'], groups=data_new['treatments'], alpha=0.05)

print(m_comp)
w, pvalue = stats.shapiro(model.resid)

print(w, pvalue)
w,pval=stats.bartlett(datat['mba_p'],datat['Comm&Mgmt'],datat['Others'],datat['Sci&Tech'])

print(w,pval)