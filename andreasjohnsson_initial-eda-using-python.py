# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 





import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import scipy.stats as stats 

from sklearn import preprocessing





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
# Basic properties of the TRAIN data set

train.describe()
# Basic properties of the TEST data set

test.describe()
contIndex = [(("cont" in x) or ("loss" in x)) for x in train.columns]

catIndex = [(("cat" in x) or ("loss" in x)) for x in train.columns]

# Plot heatmap for continuous features



sns.set(context="paper", font="monospace")



corr = train.loc[:, contIndex].corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



sns.heatmap(corr, cmap=cmap, vmax=.3, #mask=mask,

            square=True, 

            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)

# Pair plot to understand distributions and pair-wise correlation



pltset = train.loc[:,contIndex].sample(100000)



pltset['loss75threshold'] = (pltset['loss']>3864.045).astype(int)



sns.pairplot(pltset, diag_kind="kde", hue="loss75threshold", palette="husl")



# Understand the categorial variables. Change y to see the categorial variable

pltset_cat = train.loc[:,catIndex].sample(100000)



pltset_cat['logloss'] = np.log(pltset_cat['loss'])



pltset_cat['loss75threshold'] = (pltset_cat['loss']>3864.045).astype(int)



#sns.boxplot(x="logloss", y="cat5", data=pltset_cat, hue="loss75threshold");



sns.boxplot(x="logloss", y="cat115", data=pltset_cat);



# Count the number of categories in each CAT feature



catIndex_withoutLoss = [(("cat" in x)) for x in train.columns]



tmp = []



for x in train.columns[catIndex_withoutLoss]:

    tmp.append(len(set(train.loc[:, x])))



catHist = pd.DataFrame()

catHist['count'] = tmp



catHist.plot.bar()
