# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
haberman = pd.read_csv("../input/haberman.csv/haberman.csv")
# Checking the shape of dataframe to determine how many data points and feature are present. 
print (haberman.shape)
# Looking at the column names to determine the classes of our dataframe.
print (haberman.columns)
# Checking how many data points for each class are present.
haberman["status"].value_counts()
haber_one = haberman[haberman['status'] == 1]
haber_two = haberman[haberman['status'] == 2]
label = ['1','2']
plt.plot(haber_one["age"], np.zeros_like(haber_one["age"]), 'o')
plt.plot(haber_two["age"], np.zeros_like(haber_two["age"]), 'o')
plt.title("1-D scatter plot for Age")
plt.xlabel("Age")
plt.legend(label)
plt.show()
haberman.plot(kind='scatter', x='age', y='nodes') 
plt.show()
sns.FacetGrid(haberman, hue="status", height=5).map(plt.scatter, "age", "nodes").add_legend()
plt.show()
sns.pairplot(haberman, hue="status", height=4, vars=['age', 'year', 'nodes'])
plt.show()
sns.FacetGrid(haberman, hue="status", height=5).map(sns.distplot, "age").add_legend()
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Density')
plt.show()
sns.FacetGrid(haberman, hue="status", height=5).map(sns.distplot, "year").add_legend()
plt.title('Histogram of Year')
plt.xlabel('Year')
plt.ylabel('Density')
plt.show()
sns.FacetGrid(haberman, hue="status", height=5).map(sns.distplot, "nodes").add_legend()
plt.title('Histogram of Nodes')
plt.xlabel('Nodes')
plt.ylabel('Density')
plt.show()
label = ["pdf of status 1", "cdf of status 1", "pdf of status 2", "cdf of status 2"]
counts, bin_edges = np.histogram(haber_one["age"], bins=10, density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.title("pdf and cdf for Age")
plt.xlabel("Age")
plt.ylabel("% of person's")
plt.plot(bin_edges[1:], pdf)
plt.plot(bin_edges[1:], cdf)

counts, bin_edges = np.histogram(haber_two["age"], bins=10, density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf)
plt.plot(bin_edges[1:], cdf)
plt.legend(label)

plt.show()
label = ["pdf of status 1", "cdf of status 1", "pdf of status 2", "cdf of status 2"]
counts, bin_edges = np.histogram(haber_one["year"], bins=10, density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.title("pdf and cdf for Year")
plt.xlabel("Year")
plt.ylabel("% of person's")
plt.plot(bin_edges[1:], pdf)
plt.plot(bin_edges[1:], cdf)

counts, bin_edges = np.histogram(haber_two["year"], bins=10, density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf)
plt.plot(bin_edges[1:], cdf)
plt.legend(label)

plt.show()
label = ["pdf of status 1", "cdf of status 1", "pdf of status 2", "cdf of status 2"]
counts, bin_edges = np.histogram(haber_one["nodes"], bins=10, density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.title("pdf and cdf for Nodes")
plt.xlabel("Nodes")
plt.ylabel("% of person's")
plt.plot(bin_edges[1:], pdf)
plt.plot(bin_edges[1:], cdf)

counts, bin_edges = np.histogram(haber_two["nodes"], bins=10, density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf)
plt.plot(bin_edges[1:], cdf)
plt.legend(label)

plt.show()
sns.boxplot(x='status', y='age', data=haberman).set_title("Box plot for survival_status and age")
plt.show()
sns.boxplot(x='status', y='year', data=haberman).set_title("Box plot for survival_status and Year")
plt.show()
sns.boxplot(x='status', y='nodes', data=haberman).set_title("Box plot for survival_status and Nodes")
plt.show()
sns.violinplot(x='status', y='age', data=haberman).set_title("Violin plot for survival_status and Age")
plt.show()
sns.violinplot(x='status', y='year', data=haberman).set_title("Box plot for survival_status and Year")
plt.show()
sns.violinplot(x='status', y='nodes', data=haberman).set_title("Box plot for survival_status and Nodes")
plt.show()
#2D Density plot, contours-plot
sns.jointplot(x="age", y="year", data=haber_one, kind="kde")
plt.show()
sns.jointplot(x="age", y="nodes", data=haber_one, kind="kde")
plt.show()
sns.jointplot(x="nodes", y="year", data=haber_one, kind="kde")
plt.show()