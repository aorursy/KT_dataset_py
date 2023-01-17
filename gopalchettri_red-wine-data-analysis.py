# importing required libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# loading wine dataset

red_wine_df = pd.read_csv("../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")
# Checking Top 5 records

red_wine_df.head()
# Checking columns

red_wine_df.columns
# Displaying the Details of the dataset

red_wine_df.info()
red_wine_df.describe()
# Showing the Quality of Wine

red_wine_df['quality']
red_wine_df.describe()
sns.set(rc={'figure.figsize':(7,6)})

sns.countplot(red_wine_df['quality'])
sns.pairplot(red_wine_df)
sns.set(rc={'figure.figsize':(12,10)})

sns.heatmap(red_wine_df.corr(), annot=True, fmt='.2f', linewidths=2)
sns.distplot(red_wine_df['alcohol'])
from scipy.stats import skew

skew(red_wine_df['alcohol'])
def draw_hist(temp_df, bin_size = 15):

    ax = sns.distplot(temp_df)

    #xmin, xmax = ax.get_xlim()

    #ax.set_xticks(np.round(np.linspace(xmin, xmax, bin_size), 2))

    plt.tight_layout()

    plt.locator_params(axis='y', nbins=6)

    plt.show()

    print("Skewness is {}".format(skew(temp_df)))

    print("Mean is {}".format(np.median(temp_df)))

    print("Median is {}".format(np.mean(temp_df)))
draw_hist(red_wine_df['alcohol'])
sns.boxplot(x='quality', y='alcohol', data=red_wine_df)
sns.boxplot(x='quality', y='alcohol', data=red_wine_df)
joint_plt = sns.jointplot(x='alcohol', y='pH', data=red_wine_df,

                        kind='reg')
from scipy.stats import pearsonr

def get_corr(col1, col2, temp_df):

    pearson_corr, p_value = pearsonr(temp_df[col1], temp_df[col2])

    print("Correlation between {} and {} is {}".format(col1, col2, pearson_corr))

    print("P-value of this correlation is {}".format(p_value))
get_corr('alcohol', 'pH', red_wine_df)
oint_plt = sns.jointplot(x='alcohol', y='density', data=red_wine_df,

                        kind='reg')
get_corr('alcohol', 'density', red_wine_df)
g = sns.FacetGrid(red_wine_df, col="quality")

g = g.map(sns.regplot, "density", "alcohol")
sns.boxplot(x='quality', y='sulphates', data=red_wine_df)
sns.boxplot(x='quality', y='total sulfur dioxide', data=red_wine_df)
sns.boxplot(x='quality', y='free sulfur dioxide', data=red_wine_df)
red_wine_df.columns
sns.boxplot(x='quality', y='fixed acidity', data=red_wine_df)
sns.boxplot(x='quality', y='citric acid', data=red_wine_df)
sns.boxplot(x='quality', y='volatile acidity', data=red_wine_df)
red_wine_df.columns
get_corr('pH', 'citric acid', red_wine_df)
red_wine_df['total acidity'] = (red_wine_df['fixed acidity']+ red_wine_df['citric acid'] + red_wine_df['volatile acidity'])

sns.boxplot(x='quality', y='total acidity', data=red_wine_df,

           showfliers=False)
sns.regplot(x='pH', y='total acidity', data=red_wine_df)
g = sns.FacetGrid(red_wine_df, col="quality")

g = g.map(sns.regplot, "total acidity", "pH")
get_corr('total acidity', 'pH', red_wine_df)
g = sns.FacetGrid(red_wine_df, col="quality")

g = g.map(sns.regplot, "free sulfur dioxide", "pH")