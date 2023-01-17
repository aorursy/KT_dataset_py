# Supress Warnings

import warnings

warnings.filterwarnings('ignore')
# Importing libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# visulaisation

from matplotlib.pyplot import xticks

%matplotlib inline
# Data display coustomization

pd.set_option('display.max_columns', None)

pd.set_option('display.max_colwidth', -1)
# To perform Hierarchical clustering

from scipy.cluster.hierarchy import linkage

from scipy.cluster.hierarchy import dendrogram

from scipy.cluster.hierarchy import cut_tree

from sklearn.metrics import silhouette_score

from sklearn.cluster import KMeans
# import all libraries and dependencies for machine learning

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.decomposition import IncrementalPCA

from sklearn.neighbors import NearestNeighbors

from random import sample

from numpy.random import uniform

from math import isnan
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
ngo= pd.read_csv(r"/kaggle/input/help-international/Country-data.csv")

ngo.head()
word=pd.read_csv(r"/kaggle/input/help-international/data-dictionary.csv")

word.head(len(word))
ngo_dub = ngo.copy()



# Checking for duplicates and dropping the entire duplicate row if any

ngo_dub.drop_duplicates(subset=None, inplace=True)

ngo_dub.shape
ngo.shape
ngo.shape
ngo.info()
ngo.describe()
(ngo.isnull().sum() * 100 / len(ngo)).value_counts(ascending=False)
ngo.isnull().sum().value_counts(ascending=False)
(ngo.isnull().sum(axis=1) * 100 / len(ngo)).value_counts(ascending=False)
ngo.isnull().sum(axis=1).value_counts(ascending=False)
# Child Mortality Rate : Death of children under 5 years of age per 1000 live births

plt.figure(figsize = (30,5))

child_mort = ngo[['country','child_mort']].sort_values('child_mort', ascending = False)

ax = sns.barplot(x='country', y='child_mort', data= child_mort)

ax.set(xlabel = '', ylabel= 'Child Mortality Rate')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize = (10,5))

child_mort_top10 = ngo[['country','child_mort']].sort_values('child_mort', ascending = False).head(10)

ax = sns.barplot(x='country', y='child_mort', data= child_mort_top10)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set(xlabel = '', ylabel= 'Child Mortality Rate')

plt.xticks(rotation=90)

plt.show()
# Fertility Rate: The number of children that would be born to each woman if the current age-fertility rates remain the same

plt.figure(figsize = (30,5))

total_fer = ngo[['country','total_fer']].sort_values('total_fer', ascending = False)

ax = sns.barplot(x='country', y='total_fer', data= total_fer)

ax.set(xlabel = '', ylabel= 'Fertility Rate')

plt.xticks(rotation=90)

plt.show()

plt.figure(figsize = (10,5))

total_fer_top10 = ngo[['country','total_fer']].sort_values('total_fer', ascending = False).head(10)

ax = sns.barplot(x='country', y='total_fer', data= total_fer_top10)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set(xlabel = '', ylabel= 'Fertility Rate')

plt.xticks(rotation=90)

plt.show()
# Life Expectancy: The average number of years a new born child would live if the current mortality patterns are to remain same

plt.figure(figsize = (32,5))

life_expec = ngo[['country','life_expec']].sort_values('life_expec', ascending = True)

ax = sns.barplot(x='country', y='life_expec', data= life_expec)

ax.set(xlabel = '', ylabel= 'Life Expectancy')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize = (10,5))

life_expec_bottom10 = ngo[['country','life_expec']].sort_values('life_expec', ascending = True).head(10)

ax = sns.barplot(x='country', y='life_expec', data= life_expec_bottom10)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set(xlabel = '', ylabel= 'Life Expectancy')

plt.xticks(rotation=90)

plt.show()
# Health :Total health spending as %age of Total GDP.

plt.figure(figsize = (32,5))

health = ngo[['country','health']].sort_values('health', ascending = True)

ax = sns.barplot(x='country', y='health', data= health)

ax.set(xlabel = '', ylabel= 'Health')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize = (10,5))

health_bottom10 = ngo[['country','health']].sort_values('health', ascending = True).head(10)

ax = sns.barplot(x='country', y='health', data= health_bottom10)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set(xlabel = '', ylabel= 'Health')

plt.xticks(rotation=90)

plt.show()
# The GDP per capita : Calculated as the Total GDP divided by the total population.

plt.figure(figsize = (32,5))

gdpp = ngo[['country','gdpp']].sort_values('gdpp', ascending = True)

ax = sns.barplot(x='country', y='gdpp', data= gdpp)

ax.set(xlabel = '', ylabel= 'GDP per capita')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize = (10,5))

gdpp_bottom10 = ngo[['country','gdpp']].sort_values('gdpp', ascending = True).head(10)

ax = sns.barplot(x='country', y='gdpp', data= gdpp_bottom10)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set(xlabel = '', ylabel= 'GDP per capita')

plt.xticks(rotation=90)

plt.show()
# Per capita Income : Net income per person

plt.figure(figsize = (32,5))

income = ngo[['country','income']].sort_values('income', ascending = True)

ax = sns.barplot(x='country', y='income', data=income)

ax.set(xlabel = '', ylabel= 'Per capita Income')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize = (10,5))

income_bottom10 = ngo[['country','income']].sort_values('income', ascending = True).head(10)

ax = sns.barplot(x='country', y='income', data= income_bottom10)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set(xlabel = '', ylabel= 'Per capita Income')

plt.xticks(rotation=90)

plt.show()
# Inflation: The measurement of the annual growth rate of the Total GDP

plt.figure(figsize = (32,5))

inflation = ngo[['country','inflation']].sort_values('inflation', ascending = False)

ax = sns.barplot(x='country', y='inflation', data= inflation)

ax.set(xlabel = '', ylabel= 'Inflation')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize = (10,5))

inflation_top10 = ngo[['country','inflation']].sort_values('inflation', ascending = False).head(10)

ax = sns.barplot(x='country', y='inflation', data= inflation_top10)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set(xlabel = '', ylabel= 'Inflation')

plt.xticks(rotation=90)

plt.show()
# Exports: Exports of goods and services. Given as %age of the Total GDP

plt.figure(figsize = (32,5))

exports = ngo[['country','exports']].sort_values('exports', ascending = True)

ax = sns.barplot(x='country', y='exports', data= exports)

ax.set(xlabel = '', ylabel= 'Exports')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize = (10,5))

exports_bottom10 = ngo[['country','exports']].sort_values('exports', ascending = True).head(10)

ax = sns.barplot(x='country', y='exports', data= exports_bottom10)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set(xlabel = '', ylabel= 'Exports')

plt.xticks(rotation=90)

plt.show()
# Imports: Imports of goods and services. Given as %age of the Total GDP

plt.figure(figsize = (32,5))

imports = ngo[['country','imports']].sort_values('imports', ascending = True)

ax = sns.barplot(x='country', y='imports', data= imports)

ax.set(xlabel = '', ylabel= 'Imports')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize = (10,5))

imports_bottom10 = ngo[['country','imports']].sort_values('imports', ascending = True).head(10)

ax = sns.barplot(x='country', y='imports', data= imports_bottom10)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set(xlabel = '', ylabel= 'Imports')

plt.xticks(rotation=90)

plt.show()
fig, axs = plt.subplots(3,3,figsize = (18,18))



# Child Mortality Rate : Death of children under 5 years of age per 1000 live births



top5_child_mort = ngo[['country','child_mort']].sort_values('child_mort', ascending = False).head()

ax = sns.barplot(x='country', y='child_mort', data= top5_child_mort, ax = axs[0,0])

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set(xlabel = '', ylabel= 'Child Mortality Rate')



# Fertility Rate: The number of children that would be born to each woman if the current age-fertility rates remain the same

top5_total_fer = ngo[['country','total_fer']].sort_values('total_fer', ascending = False).head()

ax = sns.barplot(x='country', y='total_fer', data= top5_total_fer, ax = axs[0,1])

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set(xlabel = '', ylabel= 'Fertility Rate')



# Life Expectancy: The average number of years a new born child would live if the current mortality patterns are to remain same



bottom5_life_expec = ngo[['country','life_expec']].sort_values('life_expec', ascending = True).head()

ax = sns.barplot(x='country', y='life_expec', data= bottom5_life_expec, ax = axs[0,2])

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set(xlabel = '', ylabel= 'Life Expectancy')



# Health :Total health spending as %age of Total GDP.



bottom5_health = ngo[['country','health']].sort_values('health', ascending = True).head()

ax = sns.barplot(x='country', y='health', data= bottom5_health, ax = axs[1,0])

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set(xlabel = '', ylabel= 'Health')



# The GDP per capita : Calculated as the Total GDP divided by the total population.



bottom5_gdpp = ngo[['country','gdpp']].sort_values('gdpp', ascending = True).head()

ax = sns.barplot(x='country', y='gdpp', data= bottom5_gdpp, ax = axs[1,1])

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set(xlabel = '', ylabel= 'GDP per capita')



# Per capita Income : Net income per person



bottom5_income = ngo[['country','income']].sort_values('income', ascending = True).head()

ax = sns.barplot(x='country', y='income', data= bottom5_income, ax = axs[1,2])

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set(xlabel = '', ylabel= 'Per capita Income')





# Inflation: The measurement of the annual growth rate of the Total GDP



top5_inflation = ngo[['country','inflation']].sort_values('inflation', ascending = False).head()

ax = sns.barplot(x='country', y='inflation', data= top5_inflation, ax = axs[2,0])

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set(xlabel = '', ylabel= 'Inflation')





# Exports: Exports of goods and services. Given as %age of the Total GDP



bottom5_exports = ngo[['country','exports']].sort_values('exports', ascending = True).head()

ax = sns.barplot(x='country', y='exports', data= bottom5_exports, ax = axs[2,1])

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set(xlabel = '', ylabel= 'Exports')





# Imports: Imports of goods and services. Given as %age of the Total GDP



bottom5_imports = ngo[['country','imports']].sort_values('imports', ascending = True).head()

ax = sns.barplot(x='country', y='imports', data= bottom5_imports, ax = axs[2,2])

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set(xlabel = '', ylabel= 'Imports')



for ax in fig.axes:

    plt.sca(ax)

    plt.xticks(rotation = 90)

    

plt.tight_layout()

plt.savefig('EDA')

plt.show()
# Let's check the correlation coefficients to see which variables are highly correlated



plt.figure(figsize = (10, 10))

sns.heatmap(ngo.corr(), annot = True, cmap="rainbow")

plt.savefig('Correlation')

plt.show()
sns.pairplot(ngo,corner=True,diag_kind="kde")

plt.show()
# Converting exports,imports and health spending percentages to absolute values.



ngo['exports'] = ngo['exports'] * ngo['gdpp']/100

ngo['imports'] = ngo['imports'] * ngo['gdpp']/100

ngo['health'] = ngo['health'] * ngo['gdpp']/100
ngo.head()
import pandas_profiling

pandas_profiling.ProfileReport(ngo)