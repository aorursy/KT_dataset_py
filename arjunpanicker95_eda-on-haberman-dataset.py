# Import necessary libraries

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np



sns.set_style('darkgrid')
cols = ['age', 'op_year', 'axil_nodes', 'surv_status']

data = pd.read_csv('../input/habermans-survival-data-set/haberman.csv', names=cols)

data.info()
data.surv_status.value_counts()
data.describe()
data['survived_more_than_5yrs'] = data['surv_status'].apply(lambda x: 'yes' if x == 1 else 'no')

data.drop('surv_status', axis=1, inplace=True)

data.head()
data.info()
for indx, feature in enumerate(data.columns[:-1]):

    fig = sns.FacetGrid(data, hue='survived_more_than_5yrs', height=5)

    fig.map(sns.distplot, feature).add_legend()

    plt.show()
def get_cdf_info(dataset):

    ''' Calculate pdf, cdf and bins for the dataset '''

    cdf = {}

    pdf = {}

    bins = {}

    for indx, feature in enumerate(dataset.columns[:-1]):

        counts, bins_edge = np.histogram(dataset[feature], bins=15, density=True)

        pdf[feature] = np.asarray(counts / np.sum(counts))

        cdf[feature] = np.cumsum(pdf[feature])

        bins[feature] = bins_edge

        

    return pdf, cdf, bins
# Divide the dataset based on the survival status of the patients

survived_data = data[data['survived_more_than_5yrs'] == 'yes'].copy()

not_survived_data = data[data['survived_more_than_5yrs'] == 'no'].copy()
# Plotting the cdf and pdf for the survived dataset

pdf, cdf, bins = get_cdf_info(survived_data.copy())



plt.figure(figsize=(20, 5))

for indx, feature in enumerate(survived_data.columns[:-1]):

    plt.subplot(1, 3, indx + 1)

    fig = plt.plot(bins[feature][1:], cdf[feature])

    plt.plot(bins[feature][1:], pdf[feature])

    plt.legend(['PDF', 'CDF'])

    plt.xlabel(feature)

    plt.ylabel('Probability')
# Plotting the cdf and pdf for the not survived dataset

pdf, cdf, bins = get_cdf_info(not_survived_data.copy())



plt.figure(figsize=(20, 5))

for indx, feature in enumerate(not_survived_data.columns[:-1]):

    plt.subplot(1, 3, indx + 1)

    fig = plt.plot(bins[feature][1:], cdf[feature])

    plt.plot(bins[feature][1:], pdf[feature])

    plt.legend(['PDF', 'CDF'])

    plt.xlabel(feature)

    plt.ylabel('Probability')
# Box plots

plt.figure(figsize=(15, 5))

for indx, feature in enumerate(data.columns[:-1]):

    plt.subplot(1, 3, indx + 1)

    plt.subplots_adjust(wspace=0.8)

    sns.boxplot(x='survived_more_than_5yrs', y=feature, data=data)
# Violin plots - Shows the pdf on top of the basic box plot

plt.figure(figsize=(15, 5))

for indx, feature in enumerate(data.columns[:-1]):

    plt.subplot(1, 3, indx + 1)

    plt.subplots_adjust(wspace=0.8)

    sns.violinplot(x='survived_more_than_5yrs', y=feature, data=data)
sns.pairplot(data, hue='survived_more_than_5yrs', height=5)

plt.show()
sns.scatterplot(x='age', y='axil_nodes', data=data, hue='survived_more_than_5yrs')

plt.show()