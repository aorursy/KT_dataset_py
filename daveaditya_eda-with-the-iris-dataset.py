!pip install seaborn --upgrade
# Import libraries for Data Analysis

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import datasets



%matplotlib inline



# Set default figure size

sns.set(rc={'figure.figsize': (16, 8)})



# Set color palette for all graphs

sns.color_palette("coolwarm");
iris_ds = datasets.load_iris()
iris_ds
# Create dataframe from the dataset

df = pd.DataFrame(iris_ds['data'], columns=iris_ds['feature_names'])



# Show the first five records, to know that we have correctly created the DataFrame

df.head()
# Add the 'Target' column to the DataFrame

df['target'] = iris_ds['target']
df.head()
print(iris_ds['DESCR'])
df.shape
df.dtypes
df.info()
df.describe()
df.corr()
# Map the target column to respective Iris species

target = df['target'].map({0: 'Setosa', 1: 'Versicolour', 2: 'Virginica'})
fig, axes = plt.subplots(2, 2, figsize=(16, 8), sharex=False)

fig.suptitle('Distribution of the Features')



sns.histplot(df['sepal length (cm)'], ax=axes[0, 0], kde=True)

sns.histplot(df['sepal width (cm)'], ax=axes[0, 1], kde=True)

sns.histplot(df['petal length (cm)'], ax=axes[1, 0], kde=True)

sns.histplot(df['petal width (cm)'], ax=axes[1, 1], kde=True);
# Create y with numbering

numbers = np.arange(df.shape[0])



fig, axes = plt.subplots(2, 2, figsize=(16, 8), sharex=False)

fig.suptitle('Visualizing clusters based on the Species')



sns.scatterplot(data=df, x='sepal length (cm)', y=numbers, hue=target, ax=axes[0, 0])

sns.scatterplot(data=df, x='sepal width (cm)', y=numbers, hue=target, ax=axes[0, 1])

sns.scatterplot(data=df, x='petal length (cm)', y=numbers, hue=target, ax=axes[1, 0])

sns.scatterplot(data=df, x='petal width (cm)', y=numbers, hue=target, ax=axes[1, 1]);
fig, axes = plt.subplots(2, 2, figsize=(16, 8), sharex=False)

fig.suptitle('Plotting the features against target ("species")')



sns.scatterplot(data=df, x='sepal length (cm)', y='target', hue=target, ax=axes[0, 0])

sns.scatterplot(data=df, x='sepal width (cm)', y='target', hue=target, ax=axes[0, 1])

sns.scatterplot(data=df, x='petal length (cm)', y='target', hue=target, ax=axes[1, 0])

sns.scatterplot(data=df, x='petal width (cm)', y='target', hue=target, ax=axes[1, 1]);
fig, axes = plt.subplots(2, 2, figsize=(16, 8), sharex=False)

fig.suptitle('Plotting the features against target ("species")')



sns.lineplot(data=df, x='sepal length (cm)', y='target', hue=target, ax=axes[0, 0])

sns.lineplot(data=df, x='sepal width (cm)', y='target', hue=target, ax=axes[0, 1])

sns.lineplot(data=df, x='petal length (cm)', y='target', hue=target, ax=axes[1, 0])

sns.lineplot(data=df, x='petal width (cm)', y='target', hue=target, ax=axes[1, 1]);
# Create y with numbering

numbers = np.arange(df.shape[0])



fig, axes = plt.subplots(2, 2, figsize=(16, 8), sharex=False)

fig.suptitle('Distribution of the Features')



sns.lineplot(data=df, x='sepal length (cm)', y=numbers, hue=target, ax=axes[0, 0])

sns.lineplot(data=df, x='sepal width (cm)', y=numbers, hue=target, ax=axes[0, 1])

sns.lineplot(data=df, x='petal length (cm)', y=numbers, hue=target, ax=axes[1, 0])

sns.lineplot(data=df, x='petal width (cm)', y=numbers, hue=target, ax=axes[1, 1]);
sns.pairplot(data=df, hue='target', palette='Set2');
sns.set(rc={'figure.figsize': (16, 8)})

sns.boxplot(data=df.drop(['target'], axis=1))

plt.title('Statistical Analysis of Features');
sns.violinplot(data=df.drop(['target'], axis=1));
sns.swarmplot(data=df.drop(['target'], axis=1));
sns.violinplot(data=df.drop(['target'], axis=1))

sns.swarmplot(data=df.drop(['target'], axis=1), color='white');
fig, axes = plt.subplots(2, 2, figsize=(20, 10))

fig.suptitle('Distribution of Features based on Category')



sns.boxplot(data=df, x='sepal length (cm)', y=target, hue=target, ax=axes[0, 0])

sns.boxplot(data=df, x='sepal width (cm)', y=target, hue=target, ax=axes[0, 1])

sns.boxplot(data=df, x='petal length (cm)', y=target, hue=target, ax=axes[1, 0])

sns.boxplot(data=df, x='petal width (cm)', y=target, hue=target, ax=axes[1, 1]);
fig, axes = plt.subplots(2, 2, figsize=(20, 15))

fig.suptitle('Distribution of Features based on Category')



sns.violinplot(data=df, x='sepal length (cm)', y=target, hue=target, ax=axes[0, 0])

sns.violinplot(data=df, x='sepal width (cm)', y=target, hue=target, ax=axes[0, 1])

sns.violinplot(data=df, x='petal length (cm)', y=target, hue=target, ax=axes[1, 0])

sns.violinplot(data=df, x='petal width (cm)', y=target, hue=target, ax=axes[1, 1]);
# Creating separate datasets based on species

df_setosa = df[df['target'] == 0].drop(['target'], axis=1)

df_versicolour = df[df['target'] == 1].drop(['target'], axis=1)

df_virginica = df[df['target'] == 2].drop(['target'], axis=1)
fig, axes = plt.subplots(1, 4, figsize=(16, 8))

fig.suptitle('Distribution of Features for Iris-Setosa')



# First Plot

sns.histplot(data=df_setosa, x='sepal length (cm)', kde=True, ax=axes[0])



# Second Plot

sns.histplot(data=df_setosa, x='sepal width (cm)', kde=True, ax=axes[1])



# Third Plot

sns.histplot(data=df_setosa, x='petal length (cm)', kde=True, ax=axes[2])



# Fourth Plot

sns.histplot(data=df_setosa, x='petal width (cm)', kde=True, ax=axes[3]);
fig, axes = plt.subplots(1, 4, figsize=(16, 8))

fig.suptitle('Distribution of Features for Iris - Versicolour')



# First Plot

sns.histplot(data=df_versicolour, x='sepal length (cm)', kde=True, ax=axes[0])



# Second Plot

sns.histplot(data=df_versicolour, x='sepal width (cm)', kde=True, ax=axes[1])



# Third Plot

sns.histplot(data=df_versicolour, x='petal length (cm)', kde=True, ax=axes[2])



# Fourth Plot

sns.histplot(data=df_versicolour, x='petal width (cm)', kde=True, ax=axes[3]);
fig, axes = plt.subplots(1, 4, figsize=(16, 8))

fig.suptitle('Distribution of Features for Iris - Virginica')



# First Plot

sns.histplot(data=df_virginica, x='sepal length (cm)', kde=True, ax=axes[0])



# Second Plot

sns.histplot(data=df_virginica, x='sepal width (cm)', kde=True, ax=axes[1])



# Third Plot

sns.histplot(data=df_virginica, x='petal length (cm)', kde=True, ax=axes[2])



# Fourth Plot

sns.histplot(data=df_virginica, x='petal width (cm)', kde=True, ax=axes[3]);
sns.boxplot(data=df_setosa)

plt.title('Iris - Setosa');
sns.boxplot(data=df_versicolour)

plt.title('Iris Versicolour');
sns.boxplot(data=df_virginica)

plt.title('Iris Virginica');
fig, axes = plt.subplots(3, 1, figsize=(20, 30))



# First Plot

sns.violinplot(data=df_setosa, ax=axes[0])

axes[0].title.set_text('Iris - Setosa');



# Second Plot

sns.violinplot(data=df_versicolour, ax=axes[1])

axes[1].title.set_text('Iris - Versicolour');



# Third Plot

sns.violinplot(data=df_virginica, ax=axes[2])

axes[2].title.set_text('Iris - Virginica');
fig, axes = plt.subplots(3, 1, figsize=(20, 30))



# First Plot

sns.swarmplot(data=df_setosa, ax=axes[0], size=3)

axes[0].title.set_text('Iris - Setosa');



# Second Plot

sns.swarmplot(data=df_versicolour, ax=axes[1], size=3)

axes[1].title.set_text('Iris - Versicolour');



# Third Plot

sns.swarmplot(data=df_virginica, ax=axes[2], size=3)

axes[2].title.set_text('Iris - Virginica');
fig, axes = plt.subplots(3, 1, figsize=(20, 30))



# First Plot

sns.violinplot(data=df_setosa, ax=axes[0])

sns.swarmplot(data=df_setosa, ax=axes[0], size=3, color='white')

axes[0].title.set_text('Iris - Setosa');



# Second Plot

sns.violinplot(data=df_versicolour, ax=axes[1])

sns.swarmplot(data=df_versicolour, ax=axes[1], size=3, color='white')

axes[1].title.set_text('Iris - Versicolour');



# Third Plot

sns.violinplot(data=df_virginica, ax=axes[2])

sns.swarmplot(data=df_virginica, ax=axes[2], size=3, color='white')

axes[2].title.set_text('Iris - Virginica');
sns.pairplot(data=df_setosa, kind='reg');
sns.pairplot(data=df_versicolour, kind='reg');
sns.pairplot(data=df_virginica, kind='reg');
sns.heatmap(data=df.corr(), annot=True, cmap='coolwarm');