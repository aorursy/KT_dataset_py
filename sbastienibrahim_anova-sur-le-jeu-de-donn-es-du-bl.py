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
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import statsmodels.formula.api as smf

import statsmodels.api as sm

import os
#print(os.listdir('/kaggle/input/ble.txt'))



ble = pd.read_csv('/kaggle/input/ble.txt', sep=";", decimal='.')

#ble.head()
sns.set()



ax = sns.boxplot(x="variete", y="rdt", data=ble, color='white')

plt.xlabel('Variété de blé')

plt.ylabel('Rendement')

plt.title('Boîtes à moustaches')

plt.show()
anova_variete = smf.ols('rdt~variete', data=ble).fit()

print(anova_variete.summary())
sm.stats.anova_lm(anova_variete, typ=2)
ax = sns.boxplot(x="phyto", y="rdt", data=ble, color='white')

plt.xlabel('Traitement phytosanitaire')

plt.ylabel('Rendement')

plt.title('Boîtes à moustaches')

plt.show()
anova_phyto = smf.ols('rdt~phyto', data=ble).fit()

print(anova_phyto.summary())

sm.stats.anova_lm(anova_phyto, typ=2)
anova_variete_phyto = smf.ols('rdt~variete*phyto', data=ble).fit()

print(anova_variete_phyto.summary())

sm.stats.anova_lm(anova_variete_phyto)