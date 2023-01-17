# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
a=np.array([1,2,3,4,5,6])

a
b=np.array([[10,20,30], [40,50,60]])

b
np.random.seed(25)

c=36*np.random.randn(6)

c
d=np.arange(1,35)

d
a*10
c+a
c-a
c*a
c/a
aa=np.array([[2,4,6],[1,3,5],[10,20,30]])

aa
bb=np.array([[0,1,2],[3,4,5],[6,7,8]])

bb
aa*bb
np.dot(aa,bb)
from scipy import stats
cars=pd.read_csv("../input/praactice-data/mt1cars.csv")

cars.head()
cars.sum()
cars.sum(axis=1)
cars.median()
cars.mean()
cars.max()
cars.min()
mpg=cars.mpg

mpg.idxmax()
cars.std()
cars.var()
cars.gear.value_counts()
cars.describe()
cars.carb.value_counts()
cars_cat=cars[['cyl','vs','am','gear', 'carb']]

cars_cat.head()
gear_group=cars_cat.groupby('gear')

gear_group.describe()
cars['group'] = cars.vs + cars.am 

cars['group'].astype('category')
cars['group'].value_counts()
pd.crosstab(cars['am'], cars['gear'])
from scipy.stats import pearsonr

from scipy.stats import spearmanr

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline 

#fig, ax = plt.subplots(1,1, figsize=(8,4))

sns.set_style('whitegrid')

x=cars[['mpg','hp','qsec', 'wt']]

sns.pairplot(x)
Pc, Pv = pearsonr(x.mpg,x.hp)

print('pearson corelation co_efficient %0.3f'% (Pc))

print('pvalue ', (Pv))
Pc, Pv = pearsonr(x.mpg,x.qsec)

print('pearson corelation co_efficient %0.3f'% (Pc))

print('pvalue ', (Pv))
Pc, Pv = pearsonr(x.mpg,x.wt)

print('pearson corelation co_efficient %0.3f'% (Pc))

print('pvalue ', (Pv))
x.corr()
sns.heatmap(x.corr(), annot=True, cmap='RdYlGn', linewidths=.5, fmt='.1f')

plt.show()
y=cars[['cyl','vs','am','gear']]

sns.pairplot(y)
Sc, Pv = pearsonr(y.cyl,y.vs)

print('spearman rank corelation co_efficient %0.3f'% (Sc))

print('pvalue ', (Pv))
Sc, Pv = pearsonr(y.cyl,y.am)

print('spearman rank corelation co_efficient %0.3f'% (Sc))

print('pvalue ', (Pv))
Sc, Pv = pearsonr(y.cyl,y.gear)

print(' spearman rank co_efficient %0.3f'% (Sc))

print('pvalue ', (Pv))
from scipy.stats import chi2_contingency

table=pd.crosstab(y.cyl,y.am)

chi2, p, dof, expected = chi2_contingency(table.values)

print(chi2)

print(p)

print(dof)

print(expected)

print("chi-square statistic %0.3f p_value %0.3f" % (chi2, p))
table=pd.crosstab(y.cyl,y.vs)

chi2, p, dof, expected = chi2_contingency(table.values)

print(chi2)

print(p)

print(dof)

print(expected)

print("chi-square statistic %0.3f p_value %0.3f" % (chi2, p))
table=pd.crosstab(y.cyl,y.am)

chi2, p, dof, expected = chi2_contingency(table.values)

print(chi2)

print(p)

print(dof)

print(expected)

print("chi-square statistic %0.3f p_value %0.3f" % (chi2, p))
table=pd.crosstab(y.cyl,y.gear)

chi2, p, dof, expected = chi2_contingency(table.values)

print(chi2)

print(p)

print(dof)

print(expected)

print("chi-square statistic %0.3f p_value %0.3f" % (chi2, p))
from sklearn.preprocessing import scale

from sklearn.preprocessing import MinMaxScaler # values between 0 and 1
cars[['mpg']].describe()
t=np.asarray(mpg)

mpg_matrix=t.reshape(-1,1)

scaled=MinMaxScaler()

scaled_mpg=scaled.fit_transform(mpg_matrix)

plt.plot(scaled_mpg)
t=np.asarray(mpg)

mpg_matrix=t.reshape(-1,1)

scaled=MinMaxScaler(feature_range=(0,10))

scaled_mpg=scaled.fit_transform(mpg_matrix)

plt.plot(scaled_mpg)
standard_mpg=scale(mpg, axis=0, with_mean=False, with_std=False) # returns back to default

plt.plot(standard_mpg)
scale_mpg=scale(mpg)

plt.plot(scale_mpg)