import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

iris = pd.read_csv('../input/Iris.csv')

iris.head()
print(iris.shape)
iris.info()
iris.describe()
iris["Species"].value_counts()
iris.columns = ['Id', 'Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width', 'Species']
print(iris.columns)
iris.plot(kind = 'scatter', x= 'Sepal_Length',y = 'Sepal_Width', figsize = (7,7))
sns.set_style('whitegrid')

fg = sns.FacetGrid(data=iris, height=6,hue='Species')

fg.map(plt.scatter, 'Sepal_Length', 'Sepal_Width').add_legend()



# sns.FacetGrid(dataset, hue = 'Species', height=4).map(plt.scatter, "Sepal_Length", "Sepal_Width").add_legend()

# plt.show()
fg = sns.FacetGrid(data=iris, height=6,hue='Species')

fg.map(plt.scatter, 'Sepal_Length', 'Sepal_Width').add_legend()



x =[4, 7.5 ]

y = [2.25, 4.5]

plt.plot(x, y, 'ro-')

plt.show()
