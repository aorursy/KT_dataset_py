# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy import stats

import statsmodels.stats.multitest as smm



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/gen.csv")

data.head()
normal = data.loc[data.Diagnosis == 'normal']

early_neoplasia = data.loc[data.Diagnosis == 'early neoplasia']

cancer = data.loc[data.Diagnosis == 'cancer']

genes = np.array(data.columns)[2:]

genes


ttest_1 = []

ttest_2 = []



for gene in genes:

    ttest_1.append(stats.ttest_ind(normal[gene], early_neoplasia[gene], equal_var = False).pvalue)

    ttest_2.append(stats.ttest_ind(early_neoplasia[gene], cancer[gene], equal_var = False).pvalue)
alpha = 0.05

ttest_1 = np.array(ttest_1)

ttest_2 = np.array(ttest_2)

print(np.sum(ttest_1 < alpha))

print(np.sum(ttest_2 < alpha))
rej_1, correct_1 = smm.multipletests(ttest_1, alpha=alpha/2, method='holm')[:2]

rej_2, correct_2 = smm.multipletests(ttest_2, alpha=alpha/2, method='holm')[:2]
def fold_change(C, T):

    res = 0

    if T > C:

        res = T/C

    elif C > T:

        res = - C/T

    return res
c_1 = 0

c_2 = 0



for i in range(len(genes)):

    gene = genes[i]

    

    if abs(fold_change(np.mean(normal[gene]), np.mean(early_neoplasia[gene]))) > 1.5 and rej_1[i]:

        c_1 += 1

    if abs(fold_change(np.mean(early_neoplasia[gene]), np.mean(cancer[gene]))) > 1.5 and rej_2[i]:

        c_2 += 1

        

c_1, c_2
rej_1, correct_1 = smm.multipletests(ttest_1, alpha=alpha/2, method='fdr_bh')[:2]

rej_2, correct_2 = smm.multipletests(ttest_2, alpha=alpha/2, method='fdr_bh')[:2]
def fold_change(C, T):

    res = 0

    if T > C:

        res = T/C

    elif C > T:

        res = - C/T

    return res
c_1 = 0

c_2 = 0



for i in range(len(genes)):

    gene = genes[i]

    

    if abs(fold_change(np.mean(normal[gene]), np.mean(early_neoplasia[gene]))) > 1.5 and rej_1[i]:

        c_1 += 1

    if abs(fold_change(np.mean(early_neoplasia[gene]), np.mean(cancer[gene]))) > 1.5 and rej_2[i]:

        c_2 += 1

        

c_1, c_2