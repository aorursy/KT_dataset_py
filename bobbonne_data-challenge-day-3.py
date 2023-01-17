# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from scipy.stats import ttest_ind

import seaborn as sns

import matplotlib.pyplot as pp
mycsv=pd.read_csv('../input/cereal.csv')

print(mycsv.head())
hot_sod=mycsv[mycsv.type=='H']['sodium']

cold_sod=mycsv[mycsv.type=='C']['sodium']

print('Sodium:\nhot:\nmean %d , std : %d' % (hot_sod.mean(),hot_sod.std()))

print('cold:\nmean %d , std : %d' % (cold_sod.mean(),cold_sod.std()))
pvalue=ttest_ind(a=hot_sod,b=cold_sod,equal_var=False).pvalue

print('pvalue=%.3f' % pvalue)
hot_sug=mycsv[mycsv.type=='H']['sugars']

cold_sug=mycsv[mycsv.type=='C']['sugars']

print('sugars:\nhot:\nmean %d , std : %d' % (hot_sug.mean(),hot_sug.std()))

print('cold:\nmean %d , std : %d' % (cold_sug.mean(),cold_sug.std()))
pvalue=ttest_ind(a=hot_sug,b=cold_sug,equal_var=False).pvalue

print('pvalue=%.3f' % pvalue)
print('Looks for pvalues on all columns...')

for col in mycsv.columns.drop(['name','mfr','type']):

    hot=mycsv[mycsv.type=='H'][col]

    cold=mycsv[mycsv.type=='C'][col]

    print('-'*40)

    print('%s:\nstd hot: %.4f , mean hot: %.4f\nstd cold: %.4f, mean cold: %.4f' % 

          (col,hot.std(),hot.mean(),cold.std(),cold.mean()))

    # equal_var to true if less than 5% diff between stds

    eq_var=True if (abs(hot.std()-cold.std())/max(hot.std(),cold.std()))<0.05 else False

    pvalue=ttest_ind(a=hot,b=cold,equal_var=eq_var).pvalue 

    print('pvalue=%.3f' % pvalue)
print('Plots for sugars and potass')

print('The mean difference in sugars is related to the cold/hot')

print('The mean difference in potass is not related to the cold/hot')

pvalues=[0.,0.]

means=[0.,0.]

fig, ax = pp.subplots(ncols=2)

fig.set_figwidth(10)

fig.set_figheight(5)

for i, col in enumerate(['sugars','potass']):

    hot=mycsv[mycsv.type=='H'][col]

    cold=mycsv[mycsv.type=='C'][col]

    # equal_var to true if less than 5% diff between stds

    eq_var=True if (abs(hot.std()-cold.std())/max(hot.std(),cold.std()))<0.05 else False

    pvalues[i]=ttest_ind(a=hot,b=cold,equal_var=eq_var).pvalue 

    means[i]=abs(hot.mean()-cold.mean())

    sns.distplot(hot,label=col+' for hot cereals\nwith mean=%.2f' % hot.mean(),kde=False,ax=ax[i])

    sns.distplot(cold,label=col+' for cold cereals\nwith mean=%.2f' % cold.mean(),kde=False,ax=ax[i],

                 axlabel='%s, with mean diff=%.2f \nand pvalue=%.2f' % (col,means[i],pvalues[i]))

    ax[i].legend()

    
