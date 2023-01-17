import pandas as pd

import seaborn as sns

import os

import numpy as np

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')
LOCATION = "../input/habermans-survival-data-set/"

data = pd.read_csv(os.path.join(LOCATION,'haberman.csv'))

data.columns=["Age","Operation_year","Axil_nodes","Surv_status"] #introudcing the coloumn names in the data

data.head()
print("The number of rows in the dataset is",data.shape[0])

print('The number of features/columns is',data.shape[1])
np.round(data.describe(),3)
print(data['Surv_status'].unique())
data.isnull().any()
data.isna().any()
"""

* Distribution plots are used to visually assess how the data points are distributed with respect to its frequency.

* Usually the data points are grouped into bins and the height of the bars representing each group increases with increase in the number of data points 

lie within that group. (histogram)

* Probality Density Function (PDF) is the probabilty that the variable takes a value x. (smoothed version of the histogram)

* Kernel Density Estimate (KDE) is the way to estimate the PDF. The area under the KDE curve is 1.

* Here the height of the bar denotes the percentage of data points under the corresponding group

"""

for idx, feature in enumerate(list(data.columns)[:-1]):

    fg = sns.FacetGrid(data, hue='Surv_status', size=5)

    fg.map(sns.distplot, feature).add_legend()

    plt.show()
"""

The cumulative distribution function (cdf) is the probability that the variable takes a value less than or equal to x.

"""

plt.figure(figsize=(20,5))

for idx, feature in enumerate(list(data.columns)[:-1]):

    plt.subplot(1, 3, idx+1)

    print("********* "+feature+" *********")

    counts, bin_edges = np.histogram(data[feature], bins=10, density=True)

    print("Bin Edges: {}".format(bin_edges))

    pdf = counts/sum(counts)

    print("PDF: {}".format(pdf))

    cdf = np.cumsum(pdf)

    print("CDF: {}".format(cdf))

    plt.plot(bin_edges[1:],pdf,bin_edges[1:], cdf)

#     plt.legend([pdf,cdf],['PDF','CDF'],loc=1)

    plt.xlabel(feature)
"""

Box plot takes a less space and visually represents the five number summary of the data points in a box. 

The outliers are displayed as points outside the box.

1. Q1 - 1.5*IQR

2. Q1 (25th percentile)

3. Q2 (50th percentile or median)

4. Q3 (75th percentile)

5. Q3 + 1.5*IQR

Inter Quartile Range = Q3 -Q1

"""

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, feature in enumerate(list(data.columns)[:-1]):

    sns.boxplot( x='Surv_status', y=feature, data=data, ax=axes[idx])

plt.show()  
"""

Violin plot is the combination of box plot and probability density function.

"""

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, feature in enumerate(list(data.columns)[:-1]):

    sns.violinplot( x='Surv_status', y=feature, data=data, ax=axes[idx])

plt.show()
# pair plot

"""

Pair plot in seaborn plots the scatter plot between every two data columns in a given dataframe.

It is used to visualize the relationship between two variables

"""

# data = data.loc[:,:-1]

sns.pairplot(data, hue='Surv_status', size=4)

# plt.scatter(data.iloc[:,:-1],data[:,-1],size=4)

# grr = pd.plottting.scatter_matrix(data, c=Y, figsize=(15, 15), marker='o',hist_kwds={'bins': 20}, s=60, alpha=.8)

plt.show()