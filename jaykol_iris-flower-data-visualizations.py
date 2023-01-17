#  From sci-kit-learn get iris dataset

#  find keys in iris dataset

from sklearn import datasets

data = datasets.load_iris()

for keys in data.keys() :

    print(keys)
#  Get iris column names

data['feature_names']
# Reformat column names

import re

new_feature = []

for feature in data['feature_names']:

    new_feature.append(re.sub(r'(\w+) (\w+) \((\w+)\)',r'\1_\2_\3',feature))

print(new_feature)
# print first 10 data values of iris dataset

data['data'][:10]
# Covert list data to Dataframe

import pandas as pd

iris = pd.DataFrame(data['data'], columns=new_feature)

iris[:10]
# Iris species

data['target_names']
# Add species column to dataframe

import numpy as np

iris['species'] = np.nan

iris['species'][:50] = 'setosa'

iris['species'][50:100] = 'versicolor'

iris['species'][100:150] = 'virginica'
# Get first 10 data of iris dataframe

iris[:10]
# Get data info to check for missing value etc.

iris.info()
# Get number of datasets in each species 

iris['species'].value_counts()
# Scatter plot for length vs width 

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(10,8))

ax1.scatter(iris['sepal_length_cm'],iris['sepal_width_cm'])

ax1.set_xlabel('sepal_length_cm')

ax1.set_ylabel('sepal_width_cm')

ax2.scatter(iris['petal_length_cm'],iris['petal_width_cm'])

ax2.set_xlabel('petal_length_cm')

ax2.set_ylabel('petal_width_cm')

ax3.scatter(iris['sepal_length_cm'],iris['petal_length_cm'])

ax3.set_xlabel('sepal_length_cm')

ax3.set_ylabel('petal_length_cm')

ax4.scatter(iris['sepal_width_cm'],iris['petal_width_cm'])

ax4.set_xlabel('sepal_width_cm')

ax4.set_ylabel('petal_width_cm')
# get univariate hist plot with bivariate scatter pot



#fig = plt.figure(figsize=(8,5))

#ax1 = fig.add_subplot(121);

#ax2 = fig.add_subplot(122);

#fig, (ax1,ax2) = plt.subplots(2)

sns.jointplot(data=iris, x='sepal_length_cm',y='sepal_width_cm')

sns.jointplot(data=iris, x='petal_length_cm',y='petal_width_cm')
# Scatter plot by species

#fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)

#fig = plt.figure()

#ax1 = fig.add_subplot(121)

#ax2 = fig.add_subplot(122)



sns.FacetGrid(data=iris, hue='species', size=5).map(plt.scatter,'sepal_length_cm','sepal_width_cm').add_legend()

sns.FacetGrid(data=iris, hue='species', size=5).map(plt.scatter,'petal_length_cm','petal_width_cm').add_legend()

sns.FacetGrid(data=iris, hue='species', size=5).map(plt.scatter,'sepal_length_cm','petal_length_cm').add_legend()

sns.FacetGrid(data=iris, hue='species', size=5).map(plt.scatter,'sepal_width_cm','petal_width_cm').add_legend()
# boxplot with jitter by species

sns.boxplot(data=iris, x='species', y='sepal_length_cm')

sns.stripplot(data=iris, x='species', y='sepal_length_cm', jitter=True, edgecolor='black')
# boxplot with jitter by species

sns.boxplot(data=iris, x='species', y='sepal_width_cm')

sns.stripplot(data=iris, x='species', y='sepal_width_cm', jitter=True, edgecolor='white')
# boxplot with jitter by species

sns.boxplot(data=iris, x='species', y='petal_length_cm')

sns.stripplot(data=iris, x='species', y='petal_length_cm', jitter=True, edgecolor='white')
# boxplot with jitter by species

sns.boxplot(data=iris, x='species', y='petal_width_cm')

sns.stripplot(data=iris, x='species', y='petal_width_cm', jitter=True, edgecolor='white')
# violin plot by species

sns.violinplot(data=iris, x='species', y='sepal_length_cm')
# violin plot by species

sns.violinplot(data=iris, x='species', y='sepal_width_cm')
# violin plot by species

sns.violinplot(data=iris, x='species', y='petal_length_cm')
# violin plot by species

sns.violinplot(data=iris, x='species', y='petal_width_cm')
# KDE plot by species

sns.FacetGrid(data=iris, hue='species').map(sns.kdeplot,'sepal_length_cm').add_legend()
# KDE plot by species

sns.FacetGrid(data=iris, hue='species').map(sns.kdeplot, 'sepal_width_cm').add_legend()
# KDE plot by species

sns.FacetGrid(data=iris, hue='species').map(sns.kdeplot, 'petal_length_cm')
# KDE plot by species

sns.FacetGrid(data=iris, hue='species').map(sns.kdeplot, 'petal_width_cm')
# pair plot with default diag_kind

sns.pairplot(data=iris, hue='species',)
# pair plot with KDE diag_kind

sns.pairplot(data=iris, hue='species', diag_kind='kde')
# boxplot by species

iris.boxplot(by='species', figsize=(20,10))
# Andrews curve by species

from pandas.tools.plotting import andrews_curves

andrews_curves(iris, 'species')
# parallel coordinates by species

from pandas.tools.plotting import parallel_coordinates

parallel_coordinates(iris,'species')
# radviz plot by species

from pandas.tools.plotting import radviz

radviz(iris,'species')