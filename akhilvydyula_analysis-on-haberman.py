import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import os

from sklearn.model_selection import train_test_split



#libraries
!ls ../input/habermans-survival-data-set/
data = pd.read_csv('../input/habermans-survival-data-set/haberman.csv',header=None,names=['age', 'year_of_treatment', 'positive_lymph_nodes', 'survival_status_after_5_years'])

data.head(10)
#wanna to change survival status colum as yes or no

data["survival_status_after_5_years"] = data["survival_status_after_5_years"].map({1:"yes",2:"no"})

data
data.describe()
data.info()
plt.figure(figsize=(20,20))

data.plot("age","year_of_treatment")

plt.show()
data["survival_status_after_5_years"].value_counts()
for idx, feature in enumerate(list(data.columns)[:-1]):

    fg = sns.FacetGrid(data, hue='survival_status_after_5_years', size=5).map(sns.distplot, feature).add_legend()

    plt.show()
#boxplot

sns.boxplot(data = data, x="survival_status_after_5_years",y="age",hue="survival_status_after_5_years")

plt.show()

sns.boxplot(data = data,x ="year_of_treatment",y= "survival_status_after_5_years",notch = True)

plt.show()
sns.violinplot(data=data,y="positive_lymph_nodes",x="survival_status_after_5_years")

plt.show()
sns.pairplot(data=data)

plt.show()