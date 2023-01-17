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



%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns



from mpl_toolkits.mplot3d import Axes3D 
df = pd.read_csv('../input/big-five-personality-test/IPIP-FFM-data-8Nov2018/data-final.csv', sep='\t')



# adjusting scores for questions with negative/opposite measurement, list of negatives taken from:



negatives = [ 

    'EXT2','EXT4','EXT6','EXT8','EXT10', # 5

    'EST2','EST4',                       # 2

    'AGR1','AGR3','AGR5','AGR7',         # 4

    'CSN2','CSN4','CSN6','CSN8',         # 4

    'OPN2','OPN4','OPN6',                # 3

]



df[negatives] = df[negatives].replace({1:5, 2:4, 3:3, 4:2, 5:1})



a=range(1,11)

ext_cols = [('EXT'+str(i)) for i in a]

est_cols = [('EST'+str(i)) for i in a]

agr_cols = [('AGR'+str(i)) for i in a]

csn_cols = [('CSN'+str(i)) for i in a]

opn_cols = [('OPN'+str(i)) for i in a]



traits = [ext_cols, est_cols, agr_cols, csn_cols, opn_cols]



df_scores = pd.DataFrame(index = df.index)



# Sum scores to calculate scores for each trait



for trait in traits:

    df_scores = pd.concat([df_scores,(df.loc[:,trait]).sum(axis=1)], axis = 1)

    

df_scores.columns=['EXT', 'EST', 'AGR', 'CSN', 'OPN']









df_scores = pd.concat([df_scores, df.country], axis = 1)



df_scores.head()




hist_plots = plt.figure(figsize=(30,20))



for i in range(5):

    plt.subplot(2,3,i+1)

    plt.title(label=df_scores.columns[i],fontsize=30)

    sns.distplot(df_scores.iloc[:,i], axlabel=False, kde=False)



hist_plots.suptitle('Histograms for each trait', fontsize=30)

plt.show()
scores_by_country = df_scores.groupby('country').mean()

scores_by_country.head()

country_distplots = plt.figure(figsize=(30,20))

for i in range(5):

    plt.subplot(2,3,i+1)

    colname = scores_by_country.columns[i]

    plt.title(label=colname,fontsize=30)

    sns.distplot(scores_by_country.iloc[:,i], axlabel=False, kde=False)



plt.show()
for trait in scores_by_country:

    print("countries with maximum average scores for ", trait)

    print(scores_by_country.loc[:,trait].sort_values()[0:5])
from scipy.stats import pearsonr



r, p = pearsonr(df_scores.EXT, df_scores.EST)

print(r,p)



df_scores.iloc[:,0:5].corr()







# Correlation values are quite low, but lets plot some scatter plots and see what it looks like:
scatter_plots = plt.figure()

sns.scatterplot(x=df_scores.AGR, y=df_scores.EXT)

plt.show()



fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

ax.scatter(df_scores.EXT, df_scores.EST, df_scores.OPN)

ax.set_xlabel('X Label')

ax.set_ylabel('Y Label')

ax.set_zlabel('Z Label')

plt.show()
import numpy as np

#count outliers in each trait





zscores = pd.DataFrame(index = df_scores.index)





for trait in list(df_scores.columns[0:5]):

    

    zscores[trait] = (df_scores[trait] - df_scores[trait].mean()) / df_scores[trait].std(ddof=0)

    

# zscores = pd.concat([zscores, df_scores.country], axis = 1)





is_neg_outlier = (zscores < -2)

count_outliers = is_neg_outlier.apply(np.count_nonzero)

count_outliers.columns=['Trait','Negative_Outliers']

print(count_outliers)



## Number of outliers per 

count_outliers.plot.bar(x='Trait', y='Negative_Outliers')



is_neg_outlier = pd.concat([is_neg_outlier, df_scores.country], axis = 1)
# Prepare data for anova, lets start with categorizing EXT





conditions = [zscores['EXT']<-2, zscores['EXT']>2]

choices = ['low', 'high']









ext_cat = pd.Series(np.select(conditions, choices, default = 'mid'))





ext_anova = pd.DataFrame(index = zscores.index)

 



# EXT_CAT.head()



ext_anova = pd.concat([ext_cat, zscores[['EST','AGR','CSN','OPN']]], axis=1)

ext_anova.columns = ['EXT_CAT','EST','AGR','CSN','OPN']

ext_anova.head()

from statsmodels.formula.api import ols



results = ols('AGR ~ C(EXT_CAT)', data=ext_anova).fit()

results.summary()
df_chisq = pd.DataFrame(index = zscores.index)

traits = ['EXT', 'EST', 'AGR', 'CSN', 'OPN']

for trait in traits:

    colname = trait + '_CAT'

    conditions = [zscores[trait]<-1.5, zscores[trait]>1.5]

    choices = ['low', 'high']

    df_chisq[colname] = pd.Series(np.select(conditions, choices, default = 'mid'), name = colname)

df_chisq.describe()



# Then perform chi square, also \\todo for now :(