# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
# print(os.listdir("../input"))

haberman = pd.read_csv("../input/haberman.csv", names = ["Age", "Operation_Year", "Axil_nodes", "Survival_status"])

# Any results you write to the current directory are saved as output.
haberman.describe()
haberman.info()
haberman["Survival_status"].value_counts()
haberman["Survival_status"].value_counts(normalize=True)
sns.set_style("whitegrid");
sns.pairplot(haberman, hue="Survival_status", size=4)
plt.legend
sns.FacetGrid(haberman, hue="Survival_status", size=5) \
   .map(sns.distplot, "Age") \
   .add_legend();
sns.FacetGrid(haberman, hue="Survival_status", size=5) \
   .map(sns.distplot, "Axil_nodes") \
   .add_legend();
sns.distplot(haberman["Axil_nodes"], kde_kws=dict(cumulative=True))
sns.boxplot(x='Survival_status',y='Axil_nodes', data=haberman)