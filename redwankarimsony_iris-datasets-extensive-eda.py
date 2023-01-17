from IPython.display import HTML, display

display(HTML("<table><tr><td><img src=https://images-na.ssl-images-amazon.com/images/I/61pLvdbjC7L._AC_.jpg></td><td><img src=https://www.fs.fed.us/wildflowers/beauty/iris/Blue_Flag/images/iris_virginica/iris_virginica_virginica_lg.jpg></td> <td><img src=https://www.plant-world-seeds.com/images/item_images/000/003/884/large_square/IRIS_VERSICOLOR.JPG?1495391088> </td></tr> <tr><td> Iris setosa</td> <td> Iris virginia</td> <td> Iris versicolor</td> </tr></table>"))
import numpy as np

import pandas as pd

from sklearn.datasets import load_iris

data = load_iris()
print(data.keys())

print(data)
data_df = pd.DataFrame(data = data.data, columns = data.feature_names )

data_df['target'] = data.target

print('Shape of the dataframe: ', data_df.shape)

data_df['species'] = None



for index in range(data_df.shape[0]):

    if data_df.iloc[index, 4] ==0:

        data_df.iloc[index, 5] = 'setosa'

    elif data_df.iloc[index, 4] ==1:

        data_df.iloc[index, 5] = 'versicolor'

    elif data_df.iloc[index, 4] ==2:

        data_df.iloc[index, 5] = 'virginica'

        



data_df.head()
data_df.target.value_counts().plot.bar()
import seaborn as sns

import matplotlib.pyplot as plt

plot_location = 'upper right'



fig = plt.figure(figsize=(10, 6))

sns.distplot(data_df['petal length (cm)'][:50], label = data.target_names[0])

sns.distplot(data_df['petal length (cm)'][50:100], label = data.target_names[1])

sns.distplot(data_df['petal length (cm)'][100:] , label = data.target_names[2] )

plt.legend(loc=plot_location)

plt.title('Distribution of Petal Lengths across 3 classes')

fig.show()
fig = plt.figure(figsize=(10, 6))

sns.distplot(data_df['petal width (cm)'][:50], label = data.target_names[0])

sns.distplot(data_df['petal width (cm)'][50:100], label = data.target_names[1])

sns.distplot(data_df['petal width (cm)'][100:] , label = data.target_names[2] )

plt.legend(loc=plot_location)

plt.title('Distribution of Petal widths across 3 classes')

fig.show()
fig = plt.figure(figsize=(10, 6))

sns.distplot(data_df['sepal length (cm)'][:50], label = data.target_names[0])

sns.distplot(data_df['sepal length (cm)'][50:100], label = data.target_names[1])

sns.distplot(data_df['sepal length (cm)'][100:] , label = data.target_names[2] )

plt.legend(loc=plot_location)

plt.title('Distribution of Sepal Lengths across 3 classes')

fig.show()
fig = plt.figure(figsize=(10, 6))

sns.distplot(data_df['sepal width (cm)'][:50], label = data.target_names[0])

sns.distplot(data_df['sepal width (cm)'][50:100], label = data.target_names[1])

sns.distplot(data_df['sepal width (cm)'][100:] , label = data.target_names[2] )

plt.legend(loc=plot_location)

plt.title('Distribution of Sepal Widths across 3 classes')

fig.show()
sns.pairplot(data_df[data.feature_names]);
iris = sns.load_dataset("iris")

sns.pairplot(iris, hue="species");
g = sns.PairGrid(data_df[data.feature_names])

g.map_diag(sns.kdeplot)

g.map_offdiag(sns.kdeplot, n_levels=6);