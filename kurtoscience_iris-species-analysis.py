# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/Iris.csv")
data.info()
data.describe()
ax1 = sns.boxplot(x="Species", y="SepalLengthCm", data=data)

ax = sns.boxplot(x="Species", y="SepalWidthCm", data=data)
ax = sns.boxplot(x="Species", y="PetalLengthCm", data=data)
ax = sns.boxplot(x="Species", y="PetalWidthCm", data=data)
plt.scatter(data.SepalLengthCm,data.PetalLengthCm, color='red',alpha=0.5,label='Length')
plt.scatter(data.SepalWidthCm,data.PetalWidthCm, color='blue',alpha=0.5,label='Width')

plt.legend()
plt.xlabel('Sepal')
plt.ylabel('Petal')
plt.title('Scatter Plot')
plt.show()
data.corr()

f,ax = plt.subplots(figsize=(15,15))
sns.heatmap(data.corr(), annot = True, linewidth=0.5, fmt = '.2f', ax=ax)

plt.show()

data.columns
data.Species.unique()
data.Species.describe()
setosa = data[data.Species == 'Iris-setosa']
versicolor = data[data.Species == 'Iris-versicolor']
virginica = data[data.Species == 'Iris-virginica']
setosa.describe()
versicolor.describe()
virginica.describe()
plt.scatter(setosa.PetalLengthCm,setosa.PetalWidthCm, color ="red", label="setosa")
plt.scatter(versicolor.PetalLengthCm,versicolor.PetalWidthCm, color ="blue", label="versicolor")
plt.scatter(virginica.PetalLengthCm,virginica.PetalWidthCm, color ="black", label="virginica")
plt.legend(loc='lower right')
plt.xlabel('PetalLengthCm')
plt.ylabel('PetalWidthCm')
plt.title('Scatter Plot')
plt.show()
plt.scatter(setosa.SepalLengthCm,setosa.SepalWidthCm, color ="red", label="setosa")
plt.scatter(versicolor.SepalLengthCm,versicolor.SepalWidthCm, color ="blue", label="versicolor")
plt.scatter(virginica.SepalLengthCm,virginica.SepalWidthCm, color ="black", label="virginica")
plt.legend(loc='lower right')
plt.xlabel('SepalLengthCm')
plt.ylabel('SepalWidthCm')
plt.title('Scatter Plot')
plt.show()
plt.hist(setosa.SepalLengthCm, color ="yellow", alpha=0.5, label="setosa")
plt.hist(versicolor.SepalLengthCm, color ="red",alpha=0.5, label="versicolor")
plt.hist(virginica.SepalLengthCm, color ="black",alpha=0.5, label="virginica")
plt.legend(loc='best')
plt.xlabel('SepalLengthCm')
plt.ylabel('Frequency')
plt.title('Histogram')
plt.show()
plt.hist(setosa.PetalLengthCm, color ="yellow", alpha=0.5, label="setosa")
plt.hist(versicolor.PetalLengthCm, color ="red",alpha=0.5, label="versicolor")
plt.hist(virginica.PetalLengthCm, color ="black",alpha=0.5, label="virginica")
plt.legend(loc='best')
plt.xlabel('PetalLengthCm')
plt.ylabel('Frequency')
plt.title('Histogram')
plt.show()