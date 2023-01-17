# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

#%matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy.stats as sp

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

df = pd.read_csv("../input/museums.csv",low_memory=False)

#df.describe()

#df.columns



df['Revenue'].fillna(0, inplace=True)

df['Revenue'].fillna(0, inplace=True)

df['norm'] = (1+df['Revenue'])/2

df['lognormRevenue'] = np.log(df['norm'])

#df.groupby(['Museum Type']).groups.keys()

df.dropna()['lognormRevenue']

print(df['lognormRevenue'])



cat1 = df[df['Museum Type']=='ZOO, AQUARIUM, OR WILDLIFE CONSERVATION']

cat2 = df[df['Museum Type']!='ZOO, AQUARIUM, OR WILDLIFE CONSERVATION']



print("Standard deviation of Zoo Revenue:")

print(np.std(cat1['lognormRevenue']))



print("\nStandard deviation of Museum Revenue: ")

print(np.std(cat2['lognormRevenue']))



print("\nT-test statistic for Revenue differences of Zoo and Musuem: ")

print(sp.ttest_ind(cat1['lognormRevenue'], cat2['lognormRevenue'], equal_var=False, nan_policy='omit'))



print("\nResult Interpretation: Null Hypothesis that no differences is rejected.\n")



fig, ax = plt.subplots(1,2)

ax[0].hist(cat1['lognormRevenue'],alpha=0.9,color='blue')

ax[1].hist(cat2.dropna()['lognormRevenue'],alpha=0.9,color='blue')

plt.title("Zoo VS Museums - Revenue")

plt.show()
