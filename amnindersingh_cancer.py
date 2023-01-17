# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
cancer = pd.read_csv('/kaggle/input/habermans-survival-data-set/haberman.csv', header=None, names=['age', 'year_of_treatment', 'positive_lymph_nodes', 'survival_status_after_5_years'])

print(cancer.head())
print(cancer.info())
print(list(cancer['survival_status_after_5_years'].unique()))
cancer['survival_status_after_5_years']=cancer['survival_status_after_5_years'].map({1:"YES", 2:"NO"})

cancer['survival_status_after_5_years']=cancer['survival_status_after_5_years'].astype('category')

print(cancer.head())
print(cancer.info())
print(cancer.describe())
cancer.shape
print("Column names :- "+ ', '.join((cancer.columns)))

print("Number of rows :", cancer.shape[0])

print("Number of columns :", cancer.shape[1])

print("Target variable distribution")

print(cancer.iloc[:,-1].value_counts())

print("*"*50)

print(cancer.iloc[:,-1].value_counts(normalize = True))
#Distribution plots

"""

* Distribution plots are used to visually assess how the data points are distributed with respect to its frequency.

* Usually the data points are grouped into bins and the height of the bars representing each group increases with increase in the number of data points 

lie within that group. (histogram)

* Probality Density Function (PDF) is the probabilty that the variable takes a value x. (smoothed version of the histogram)

* Kernel Density Estimate (KDE) is the way to estimate the PDF. The area under the KDE curve is 1.

* Here the height of the bar denotes the percentage of data points under the corresponding group

"""

for idx, feature in enumerate(list(cancer.columns)[:-1]):

    fg = sns.FacetGrid(cancer, hue='survival_status_after_5_years', height=7).map(sns.distplot, feature).add_legend()

    plt.show()
#CDF

"""

The cumulative distribution function (cdf) is the probability that the variable takes a value less than or equal to x.

"""

plt.figure(figsize=(20,5))

for idx, feature in enumerate(list(cancer.columns)[:-1]):

    plt.subplot(1, 3, idx+1)

    print("********* "+feature+" *********")

    counts, bin_edges = np.histogram(cancer[feature], bins=10, density=True)

    print("Bin Edges: {}".format(bin_edges))

    pdf = counts/sum(counts)

    print("PDF: {}".format(pdf))

    cdf = np.cumsum(pdf)

    print("CDF: {}".format(cdf))

    plt.plot(bin_edges[1:], pdf, bin_edges[1:], cdf)

    plt.xlabel(feature)
#Box Plots

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

for idx, feature in enumerate(list(cancer.columns)[:-1]):

    sns.boxplot( x='survival_status_after_5_years', y=feature, data=cancer, ax=axes[idx])

plt.show()  
#Violin Plots

"""

Violin plot is the combination of box plot and probability density function.

"""

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, feature in enumerate(list(cancer.columns)[:-1]):

    sns.violinplot( x='survival_status_after_5_years', y=feature, data=cancer, ax=axes[idx])

plt.show()
#Pair Plot

 

'''

Pair plot in seaborn plots the scatter plot between every two data columns in a given dataframe.

It is used to visualize the relationship between two variables.

'''

plt.close();

sns.set_style("whitegrid")

sns.pairplot(cancer,hue='survival_status_after_5_years',height=4)

plt.show()