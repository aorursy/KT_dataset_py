import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Any results you write to the current directory are saved as output.

iris_data = pd.read_csv('../input/Iris.csv')
# Lets take a look at this dataset



iris_data.info()

iris_data.head()
# Are the sample sizes balanced?

iris_data["Species"].value_counts()
# A pivot table might offer some insight



iris_data.pivot_table(

    values=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'], 

    index=['Species'], 

    aggfunc=np.mean)
# It looks like Iris-setosa can be identified by just looking at its petal.

# Let's confirm this

sns.FacetGrid(iris_data, hue='Species', size=6).map(sns.kdeplot, 'PetalLengthCm').add_legend()
# What other attributes might be used to differentiate the species? Let's try to find out!



sns.pairplot(iris_data.drop('Id', axis=1), hue='Species')



#sns.jointplot(x='SepalLengthCm', y='PetalLengthCm', data=iris_data, alpha=.25, color='k')
iris_filtered = iris_data[iris_data["Species"] != "Iris-setosa"]



sns.FacetGrid(iris_filtered, hue="Species", size=6).map(sns.kdeplot, "PetalLengthCm").add_legend()

sns.FacetGrid(iris_filtered, hue="Species", size=6).map(sns.kdeplot, "PetalWidthCm").add_legend()