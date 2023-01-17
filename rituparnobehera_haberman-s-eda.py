# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np



# Load Haberman dataset into panda dataframe

haberman = pd.read_csv("/kaggle/input/habermans-survival-data-set/haberman.csv", names=["age", "op_year", "axil_nodes", "surv_status"])

haberman['surv_status'] = haberman['surv_status'].map({1: True, 2: False})

haberman
# Data points and features

print("Shape of dataset: {}".format(haberman.shape))



# Column names in our haberman dataset

print("columns in dataset {}".format(haberman.columns))



# Data points for each survival status or

# People with survival status as 1/2

counts = haberman['surv_status'].value_counts()

print("Counts per category:")

print(counts)



# Its im-balanced dataset as number of datasets are different
# 2D Scatter Plot

haberman.plot(kind='scatter', x='age', y='axil_nodes')

plt.show()
# 2D scatter plot with color coding by survival status

# sns: seaborn module

sns.set_style('whitegrid');

sns.FacetGrid(haberman, hue='surv_status', height=4).map(plt.scatter, 'age', 'axil_nodes').add_legend();

plt.show();
# lets visualize each each feature.

# Can be used when number of features are high

# Cannot be used to visualize 3D, 4D or nD

# Only 2D pattern can be visualized



plt.close()

sns.set_style('whitegrid')

sns.pairplot(haberman, hue='surv_status', height=3)

plt.show()
# PDF (univariant analysis):

sns.FacetGrid(haberman, hue='surv_status', height=5).map(sns.distplot, 'axil_nodes').add_legend()

plt.show()
sns.FacetGrid(haberman, hue='surv_status', height=5).map(sns.distplot, 'op_year').add_legend()

plt.show()
sns.FacetGrid(haberman, hue='surv_status', height=5).map(sns.distplot, 'age').add_legend()

plt.show()
# lets ananlyze using CDF;

# what percentage of patient have auxiliary node less than 2



for feature in (haberman.columns[:-1]):

    counts, bin_edges = np.histogram(haberman[feature], bins = 10, density=True)

    

    #PDF

    pdf = counts/sum(counts)

    

    #CDF

    cdf = np.cumsum(pdf)

    

    print("PDF: {}".format(pdf))

    print("CDF: {}".format(cdf))

    

    plt.plot(bin_edges[1:], pdf, label="PDF")

    plt.plot(bin_edges[1:], cdf, label="CDF")

    plt.xlabel(feature)

    plt.legend()

    plt.show()
#Analysis using BOX-WHISKER PLOT

#We can summarise the following insight using the plot:

#1. Q1-1.5*IQR

#2. Q1: 25th Percentile

#3. Q2: 50th Percentile

#4. Q3: 75th Percentile

#5. Q3+1.5*IQR



fig, axes = plt.subplots(1, 3, figsize=(16,5))

for idx, feature in enumerate(haberman.columns[:-1]):

    sns.boxplot(x=haberman.columns[3], y=feature, data=haberman, ax=axes[idx])

plt.show()
fig, axes = plt.subplots(1, 3, figsize=(16,5))

for idx, feature in enumerate(haberman.columns[:-1]):

    sns.violinplot(x=haberman.columns[3], y=feature, data=haberman, ax=axes[idx])

plt.show()
#Multivariant Analysis

sns.jointplot(x= 'age',kind = 'kde', y='op_year', data = haberman)

sns.jointplot(x= 'age',kind = 'kde', y='axil_nodes', data = haberman)

plt.show()