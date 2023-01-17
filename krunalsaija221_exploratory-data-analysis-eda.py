import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# This Python 3 environment comes with many helpful analytics libraries installed

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

# provide lable to each column in dataset
col = ['Patient_age', 'Year_of_operation', 'pos_axillary_nodes', 'status']
#load dataset from csv file
dataset = pd.read_csv("../input/haberman.csv", names = col)
print(dataset.shape)
dataset['status'].value_counts()
#sample dataset -- first 5 observations
dataset.head()
# 2-D Scatter plot with color-coding for each type/class.

sns.set_style("whitegrid");
sns.FacetGrid(dataset, hue="status", size=6) \
   .map(plt.scatter, "Patient_age", "pos_axillary_nodes") \
   .add_legend();
plt.show();
sns.set_style("whitegrid");
sns.FacetGrid(dataset, hue="status", size=6) \
   .map(plt.scatter, "Year_of_operation", "pos_axillary_nodes") \
   .add_legend();
plt.show();
sns.set_style("whitegrid");
sns.pairplot(dataset, hue="status",
             vars=col[0:3])
plt.show()
sns.FacetGrid(dataset, hue="status", size=5) \
   .map(sns.distplot, "pos_axillary_nodes") \
   .add_legend();
plt.show();
sns.FacetGrid(dataset, hue="status", size=5) \
   .map(sns.distplot, "Patient_age") \
   .add_legend();
plt.show();
# alive means status=1 and dead means status =2
alive = dataset.loc[dataset['status'] == 1]
dead = dataset.loc[dataset['status'] == 2]
counts, bin_edges = np.histogram(alive['pos_axillary_nodes'], bins=20, 
                                 density = True,)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], cdf)
plt.xlabel('pos_axillary_nodes')
plt.legend(['Cdf for the patients who survive more than 5 years'])
plt.show()
counts, bin_edges = np.histogram(dead['pos_axillary_nodes'], bins=5, 
                                 density = True,)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], cdf,)
plt.legend(['Cdf for the patients who have not survived more than 5 years'])
plt.xlabel('pos_axillary_nodes')
plt.show()
print("Summary Statistics of Patients")
dataset.describe()
print("Summary Statistics of Patients, who have survived")
alive.describe()
print("Summary Statistics of Patients, who have not survived")
dead.describe()
sns.boxplot(x='status',y='pos_axillary_nodes', data=dataset)
plt.show()
sns.violinplot(x='status',y='pos_axillary_nodes', data=dataset)
plt.show()
sns.violinplot(x='status',y='Patient_age', data=dataset)
plt.show()

