# importing neccessary libraries



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
# importing the cancer survival dataset into notebook

# the columns names are not provided in the dataset, so applying the column names in the order of the column



cancer = pd.read_csv('../input/habermans-survival-data-set/haberman.csv', names=['Age', 'Operation_year', 'Axil_Nodes', 'Survival_Status'])
# general info about the dataset



cancer.info()
# top 5 data points of the dataset



cancer.head(5)
# Modifying the Target variable into more meaningful categorical column



cancer['Survival_Status'] = list(map(lambda x : 'no' if (x is 2) else 'yes', cancer['Survival_Status']))
cancer['Survival_Status']
cancer['Survival_Status'].value_counts()





# survival_status column is imbalanced by 225:81

# 225 survived after 5 years

# 81 died within 5 years
# Statistical data of the dataset



cancer.describe()

# Univariate analysis with Age, Operation year and positive axil nodes detected



for idx, feature in enumerate(list(cancer.columns)[:-1]):

    sns.set_style('whitegrid')

    c = sns.FacetGrid(data=cancer, hue='Survival_Status', size=5)

    c.map(sns.distplot, feature)

    c.add_legend()

# Distribution Plots are effective in visually assessing the datapoints

# PDF (Probability Density Function) is created by smoothing histogram values

# CDF (Cummulative Density Function) is the odds of measuring any value upto and including x



plt.figure(figsize=(20,5))

for idx, feature in enumerate(list(cancer.columns)[:-1]):

    plt.subplot(1,3,idx+1)

    print(f'\n########## {feature} ##########')

    count, bin_edges = np.histogram(cancer[feature], bins=10, density=True)

    print(f'Bin Edges: {bin_edges}')

    pdf = count/sum(count)

    print(f'PDF: {pdf}')

    cdf = np.cumsum(pdf)

    print(f'CDF: {cdf}')

    plt.plot(bin_edges[1:], pdf, bin_edges[1:], cdf)

    plt.xlabel(feature)
# Box-Plots



fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, feature in enumerate(list(cancer.columns)[:-1]):

    sns.set_style('whitegrid')

    sns.boxplot(x='Survival_Status', y=feature, data=cancer, ax=axes[idx])
# Violin-Plots



fig, axes = plt.subplots(1, 3, figsize=(15,5))

for idx, feature in enumerate(list(cancer.columns)[:-1]):

    sns.set_style('whitegrid')

    sns.violinplot(x='Survival_Status', y=feature, data=cancer, ax=axes[idx])
# Percentage of patients with axillary nodes less than or equal to 10 and survived



anp = int(cancer[(cancer['Axil_Nodes'] <= 10) & (cancer['Survival_Status'] == 'yes')].count().unique())

p = anp/len(cancer['Axil_Nodes'])*100

print(p)
# Percentage of patients with axillary nodes greater than 10 and survived



anp = int(cancer[(cancer['Axil_Nodes'] > 10) & (cancer['Survival_Status'] == 'yes')].count().unique())

p = anp/len(cancer['Axil_Nodes'])*100

print(p)
# Percentage of patients with axillary nodes less than or equal to 10 and could not survive



anp = int(cancer[(cancer['Axil_Nodes'] <= 10) & (cancer['Survival_Status'] == 'no')].count().unique())

p = anp/len(cancer['Axil_Nodes'])*100

print(p)
# Percentage of patients with axillary nodes greater than 10 and could not survive



anp = int(cancer[(cancer['Axil_Nodes'] > 10) & (cancer['Survival_Status'] == 'no')].count().unique())

p = anp/len(cancer['Axil_Nodes'])*100

print(p)
# Percentage of patients with ages between 40-60 and survived



sp = int(cancer[(cancer['Age'] > 40) & (cancer['Age'] < 60) & (cancer['Survival_Status'] == 'yes')].count().unique())

# print(f'Patient survived between age 40-60 : {sp}')

abc = (sp/len(cancer['Age']))*100

print(abc)
# Percentage of patients with ages between 40-60 and could not survive



sp = int(cancer[(cancer['Age'] > 40) & (cancer['Age'] < 60) & (cancer['Survival_Status'] == 'no')].count().unique())

# print(f'Patient survived between age 40-60 : {sp}')

abc = (sp/len(cancer['Age']))*100

print(abc)
# Bivariate Analysis

# Pairplot reveals pair-wise relationship across entire dataset



sns.pairplot(data=cancer, hue='Survival_Status')
sns.jointplot(x='Age', y='Axil_Nodes', data=cancer, kind='kde')