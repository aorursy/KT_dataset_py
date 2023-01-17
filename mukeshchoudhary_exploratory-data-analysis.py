# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
heberman= pd.read_csv("../input/habermans-survival-data-set/haberman.csv", header= None, 

                      names= ['age', 'operation_year', 'positive_lymph_nodes', 'survival_status_after_5_years'])

heberman.head()
print(heberman.shape)
heberman['survival_status_after_5_years'].value_counts()
print(heberman.iloc[:,-1].value_counts(normalize = True))
heberman['survival_status_after_5_years'].unique()
heberman.info()
heberman['survival_status_after_5_years']= heberman['survival_status_after_5_years'].map({1: 'yes', 2: 'no'})
heberman['survival_status_after_5_years']= heberman['survival_status_after_5_years'].astype('category')

heberman.head(10)
heberman.describe()
sns.set_style('whitegrid')
one= heberman.loc[heberman['survival_status_after_5_years']== 'yes']

two= heberman.loc[heberman['survival_status_after_5_years']== 'no']

plt.plot(one['age'], np.zeros_like(one['age']), 'o', label= "survival_status_after_5_years, yes")

plt.plot(two['age'], np.zeros_like(two['age']), 'o', label= "survival_status_after_5_years, no")

plt.xlabel('age')

plt.show()

sns.FacetGrid(heberman, hue= 'survival_status_after_5_years', size= 6).map(plt.scatter,'age','positive_lymph_nodes').add_legend()
sns.FacetGrid(heberman, hue= 'survival_status_after_5_years', height= 8).map(plt.scatter,'operation_year','positive_lymph_nodes').add_legend()
sns.FacetGrid(heberman, hue= 'survival_status_after_5_years', height= 5).map(sns.distplot,'age').add_legend()
sns.FacetGrid(heberman, hue= 'survival_status_after_5_years', height= 5).map(sns.distplot,'operation_year').add_legend()
sns.FacetGrid(heberman, hue= 'survival_status_after_5_years', height= 5).map(sns.distplot,'positive_lymph_nodes').add_legend()
# pdf&cdf

counts, bin_edges= np.histogram(heberman['age'], bins= 10, density= True)

pdf= counts/(sum(counts))

print(pdf)

print(bin_edges)

cdf= np.cumsum(pdf)

plt.plot(bin_edges[1:], pdf)

plt.plot(bin_edges[1:], cdf)

# pdf&cdf

counts, bin_edges= np.histogram(heberman['operation_year'], bins= 10, density= True)

pdf= counts/(sum(counts))

print(pdf)

print(bin_edges)

cdf= np.cumsum(pdf)

plt.plot(bin_edges[1:], pdf)

plt.plot(bin_edges[1:], cdf)

plt.show()

# pdf&cdf

counts, bin_edges= np.histogram(heberman['positive_lymph_nodes'], bins= 10, density= True)

pdf= counts/(sum(counts))

print(pdf)

print(bin_edges)

cdf= np.cumsum(pdf)

plt.plot(bin_edges[1:], pdf)

plt.plot(bin_edges[1:], cdf)

plt.show()
# pairwise scatter plot: Pair-Plot

# Dis-advantages: 

##Can be used when number of features are high.

##Cannot visualize higher dimensional patterns in 3-D and 4-D. 

#Only possible to view 2D patterns.

sns.pairplot(heberman, hue= 'survival_status_after_5_years', height= 3)

plt.show()
print("Means:")

print(np.mean(heberman['positive_lymph_nodes']))



print("\nStd dev.:")

print(np.std(heberman['positive_lymph_nodes']))



print("\nVariance:")

print(np.var(heberman['positive_lymph_nodes']))
print("Median:")

print(np.median(heberman['positive_lymph_nodes']))



print('\nQuantile:')

print(np.percentile(heberman['positive_lymph_nodes'], np.arange(0, 101, 25)))



#90th percentile

print("\nPercentile:")

print(np.percentile(heberman['positive_lymph_nodes'],80))



from statsmodels import robust

print("\nMedian absolute deviation:")

print(robust.mad(heberman['positive_lymph_nodes']))
#Q1- (25th percentile)

#Q2- (50th percentile or median)

#Q3- (75th percentile)

#Q4-  (100th percentile)

#Inter Quartile Range = Q3 -Q1

#whisker len- 1.5*iqr

sns.boxplot(x= 'survival_status_after_5_years', y= 'positive_lymph_nodes', data= heberman)
sns.boxplot(x= 'survival_status_after_5_years', y= 'operation_year', data= heberman)
sns.boxplot(x= 'survival_status_after_5_years', y= 'age', data= heberman)
# A violin plot combines the benefits of the previous two plots 

#and simplifies them



# Denser regions of the data are fatter, and sparser ones thinner 

#in a violin plot



sns.violinplot(x="survival_status_after_5_years", y="positive_lymph_nodes", data=heberman, size=8)

plt.show()
sns.violinplot(x="survival_status_after_5_years", y="age", data=heberman, size=8)

plt.show()
sns.violinplot(x="survival_status_after_5_years", y="operation_year", data=heberman, size=8)

plt.show()