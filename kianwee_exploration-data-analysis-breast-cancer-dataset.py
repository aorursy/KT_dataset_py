# Importing libraries

import pandas as pd

import numpy as np 

import matplotlib.pyplot as plt

import seaborn as sns  
bc_dataset = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')
bc_dataset.head()
bc_dataset.describe()
bc_dataset.isnull().sum()
# Separating the mean and looking into the data

list_mean=['diagnosis','radius_mean','texture_mean','perimeter_mean','area_mean',

      'smoothness_mean','compactness_mean','concavity_mean',

      'concave points_mean','symmetry_mean','fractal_dimension_mean']

mean_data=bc_dataset[list_mean]

mean_data.head()
# Implementing pairplot 

g = sns.pairplot(mean_data, hue = 'diagnosis')

g.map_diag(sns.distplot)

g.map_offdiag(plt.scatter)

g.add_legend()

g.fig.suptitle('FacetGrid plot (Mean)', fontsize = 20)

g.fig.subplots_adjust(top= 0.9);
# Separating the mean and looking into the data

list_se=['diagnosis','radius_se','texture_se','perimeter_se','area_se',

      'smoothness_se','compactness_se','concavity_se',

      'concave points_se','symmetry_se','fractal_dimension_se']

se_data=bc_dataset[list_se]

se_data.head()
# Implementing pairplot 

gg = sns.pairplot(se_data, hue = 'diagnosis')

gg.map_diag(sns.distplot)

gg.map_offdiag(plt.scatter)

gg.add_legend()

gg.fig.suptitle('FacetGrid plot (Standard of error)', fontsize = 20)

gg.fig.subplots_adjust(top= 0.9);
ax = sns.heatmap(mean_data.corr())
radius_diagnosis = ['diagnosis','radius_mean']

radius_corr =bc_dataset[radius_diagnosis]

radius_corr.radius_mean = radius_corr.radius_mean.round()

radius_m_corr = radius_corr[radius_corr['diagnosis'] == 'M'].groupby(['radius_mean']).size().reset_index(name = 'count')

radius_m_corr.corr()
sns.regplot(x = 'radius_mean', y = 'count', data = radius_m_corr).set_title("Mean radius vs Malignant count")
radius_b_corr = radius_corr[radius_corr['diagnosis'] == 'B'].groupby(['radius_mean']).size().reset_index(name = 'count')

radius_b_corr.corr()
sns.regplot(x = 'radius_mean', y = 'count', data = radius_b_corr).set_title("Mean radius vs Benign count")
texture_diagnosis = ['diagnosis','texture_mean']

texture_corr =bc_dataset[texture_diagnosis]

texture_corr.texture_mean = texture_corr.texture_mean.round()

texture_m_corr = texture_corr[radius_corr['diagnosis'] == 'M'].groupby(['texture_mean']).size().reset_index(name = 'count')

texture_m_corr.corr()
sns.regplot(x = 'texture_mean', y = 'count', data = texture_m_corr).set_title("Mean texture vs Malignant count")
texture_b_corr = texture_corr[texture_corr['diagnosis'] == 'B'].groupby(['texture_mean']).size().reset_index(name = 'count')

texture_b_corr.corr()
sns.regplot(x = 'texture_mean', y = 'count', data = texture_b_corr).set_title("Mean texture vs Benign count")
perimeter_diagnosis = ['diagnosis','perimeter_mean']

perimeter_corr =bc_dataset[perimeter_diagnosis]

perimeter_corr.perimeter_mean = perimeter_corr.perimeter_mean.round()

perimeter_m_corr = perimeter_corr[perimeter_corr['diagnosis'] == 'M'].groupby(['perimeter_mean']).size().reset_index(name = 'count')

perimeter_m_corr.corr()
sns.regplot(x = 'perimeter_mean', y = 'count', data = perimeter_m_corr).set_title("Mean perimeter vs Malignant count")
perimeter_b_corr = perimeter_corr[perimeter_corr['diagnosis'] == 'B'].groupby(['perimeter_mean']).size().reset_index(name = 'count')

perimeter_b_corr.corr()
sns.regplot(x = 'perimeter_mean', y = 'count', data = perimeter_b_corr).set_title("Mean perimeter vs Benign count")
compactness_diagnosis = ['diagnosis','compactness_mean']

compactness_corr =bc_dataset[compactness_diagnosis]

compactness_corr.compactness_mean = compactness_corr.compactness_mean.round(2) # Round off to 2 decimal places

compactness_m_corr = compactness_corr[compactness_corr['diagnosis'] == 'M'].groupby(['compactness_mean']).size().reset_index(name = 'count')

compactness_m_corr.corr()
sns.regplot(x = 'compactness_mean', y = 'count', data = compactness_m_corr).set_title("Mean compactness vs Malignant count")
compactness_b_corr = compactness_corr[compactness_corr['diagnosis'] == 'B'].groupby(['compactness_mean']).size().reset_index(name = 'count')

compactness_b_corr.corr()
sns.regplot(x = 'compactness_mean', y = 'count', data = compactness_b_corr).set_title("Mean compactness vs Benign count")
concavity_diagnosis = ['diagnosis','concavity_mean']

concavity_corr =bc_dataset[concavity_diagnosis]

concavity_corr.concavity_mean = concavity_corr.concavity_mean.round(2) # Round off to 2 decimal places

concavity_m_corr = concavity_corr[concavity_corr['diagnosis'] == 'M'].groupby(['concavity_mean']).size().reset_index(name = 'count')

concavity_m_corr.corr()
sns.regplot(x = 'concavity_mean', y = 'count', data = concavity_m_corr).set_title("Mean concavity vs Malignant count")
concavity_b_corr = concavity_corr[concavity_corr['diagnosis'] == 'B'].groupby(['concavity_mean']).size().reset_index(name = 'count')

concavity_b_corr.corr()
sns.regplot(x = 'concavity_mean', y = 'count', data = concavity_b_corr).set_title("Mean concavity vs Benign count")