# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
iris = pd.read_csv("/kaggle/input/iris/Iris.csv");
print(iris)
iris.plot(kind='scatter', x = 'SepalLengthCm', y ='SepalWidthCm');


import seaborn as sns
sns.FacetGrid(iris, hue = 'Species',height = 4).map(plt.scatter,"SepalLengthCm","SepalWidthCm").add_legend();
plt.close()
sns.pairplot(iris, hue="Species",height=3)
plt.show()
sns.FacetGrid(iris, hue='Species',height=5).map(sns.distplot,"PetalLengthCm").add_legend();
plt.show()
sns.FacetGrid(iris, hue='Species',height=5).map(sns.distplot,"PetalWidthCm").add_legend();
plt.show()
sns.FacetGrid(iris, hue='Species',height=5).map(sns.distplot,"SepalLengthCm").add_legend();
plt.show()
sns.FacetGrid(iris, hue='Species',height=5).map(sns.distplot,"SepalWidthCm").add_legend();
plt.show()
setosa_data = iris [iris['Species']=='Iris-setosa']
virginica_data = iris [iris['Species']=='Iris-virginica']
versicolor_data = iris [iris['Species']=='Iris-versicolor']

print("\n Mean")
print(np.mean(setosa_data['PetalLengthCm']))
print(np.mean(virginica_data['PetalLengthCm']))
print(np.mean(versicolor_data['PetalLengthCm']))

print("\n Standard Deviation")
print(np.std(setosa_data['PetalLengthCm']))
print(np.std(virginica_data['PetalLengthCm']))
print(np.std(versicolor_data['PetalLengthCm']))

print ("\n Qantiles: ")
print(np.percentile(setosa_data['PetalLengthCm'],np.arange(0,100,25)))
print(np.percentile(virginica_data['PetalLengthCm'],np.arange(0,100,25)))
print(np.percentile(versicolor_data['PetalLengthCm'],np.arange(0,100,25)))

print ("\n 90th Qantiles: ")
print(np.percentile(setosa_data['PetalLengthCm'],np.arange(0,100,90)))
print(np.percentile(virginica_data['PetalLengthCm'],np.arange(0,100,90)))
print(np.percentile(versicolor_data['PetalLengthCm'],np.arange(0,100,90)))
