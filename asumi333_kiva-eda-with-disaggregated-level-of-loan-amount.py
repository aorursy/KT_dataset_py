# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import matplotlib.cm as cm
df = pd.read_csv('../input/kiva_loans.csv')
df.head(3)
country = pd.DataFrame(df.country.value_counts())
aa = country.iloc[:10, :]
sss = aa.country
orders = np.argsort(sss)

fig = plt.figure(figsize = (15, 6))

plt.barh(range(len(orders)), sss[orders], color='skyblue', align='center')
plt.yticks(range(len(orders)), aa.index[orders])
plt.title('Top 10 countries - Number of loans')
plt.show()
labels = country.index.tolist()
sizes = country.country.tolist()
cmap = plt.cm.BuPu
colors = cmap(np.linspace(0., 1., len(labels)))
fig = plt.figure(figsize=(10,10))

plt.pie(sizes,  labels=labels, colors = colors, autopct='%1.1f%%', startangle=360)
plt.title('Number of loans by coutny - %')
plt.axis('equal')
plt.show()

# Top 5 countries
df.groupby(['country'])['loan_amount'].mean().sort_values().tail()
# Bottom 5 countries
df.groupby(['country'])['loan_amount'].mean().sort_values().head()
sector = pd.DataFrame(df.sector.value_counts())
sector.plot(kind = 'barh', color = 'skyblue', figsize = (15, 6))
plt.title('Number of loans by sector')
plt.show()
# Determine the loan level by values of each quantile
df.loan_amount.describe()
# Create levels(%)
def newvar(df):
    if df.loan_amount <= 275:
        var = 'level1'
    elif df.loan_amount > 275 and df.loan_amount <= 500 :
        var = 'level2'
    elif df.loan_amount > 500 and df.loan_amount <= 1000 :
        var = 'level3'
    elif df.loan_amount > 1000:
        var = 'level4'
    return var
df['loan_amount_level'] = df.apply(newvar, axis=1)
df2 = df.groupby(['country', 'loan_amount_level'])['id'].agg('count').unstack()
df2['sum'] = df2.sum(axis=1)
df2['lev1'] = (df2['level1']/df2['sum'])*100
df2['lev2'] = (df2['level2']/df2['sum'])*100
df2['lev3'] = (df2['level3']/df2['sum'])*100
df2['lev4'] = (df2['level4']/df2['sum'])*100
df3 = df2.fillna(0)
df4 = df3.iloc[:, 5:]
df4.head()

df5 = df4.sort_values(by = ['lev1'])
df5.plot(kind = 'bar', stacked = True, colormap = cm.Pastel1, figsize = (23, 8))
plt.title('Rate of loan size by country - pink is the rate of the smallest size of loan')
plt.show()
df6 = df4.sort_values(by = ['lev4'], ascending=False)
df6.plot(kind = 'bar', stacked = True, colormap = cm.Pastel1, figsize = (23, 8))
plt.title('Rate of loan size by country - grey is the rate of the largest size of loan')
plt.show()
mpi = pd.read_csv('../input/kiva_mpi_region_locations.csv')
mpi2 = pd.DataFrame(mpi.groupby(['country'])['MPI'].agg('mean'))
new = pd.merge(df4, mpi2, left_index = True, right_index = True)
sns.regplot(x = new.lev1, y = new.MPI, fit_reg = True)
plt.xlabel('Rate of level 1 loan')
plt.ylabel('Mean MPI')
plt.show()
new2 = new.dropna()
x = new2.lev1.reshape(-1, 1)
y = new2.MPI
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats


X2 = sm.add_constant(x)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())
sns.regplot(x = new.lev4, y = new.MPI, fit_reg = True)
plt.xlabel('Rate of level 4 loan')
plt.ylabel('Mean MPI')
plt.show()
xx = new2.lev4.reshape(-1, 1)
X2 = sm.add_constant(xx)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())