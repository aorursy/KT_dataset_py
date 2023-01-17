# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Importing the required packages



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

## Reading the Iris data



iris_data = pd.read_csv("../input/Iris.csv")
## Exploring data



#Dimensions

iris_data.shape
# Columns



iris_data.columns
# Head of Data



iris_data.head()
# Tail of Data



iris_data.tail()
## Checking for null values in the data



iris_data.isnull().values.any()
iris_data
# Retrieving the Unique values 



iris_data.Species.unique()
## Segregationg the data according to Species



# Iris Setosa

iris_data1 = iris_data[iris_data['Species'] == 'Iris-setosa']

iris_data1.head()
## Taking a look at the distributions and density of Iris-Setosa



hist1 = sns.distplot(iris_data1['SepalLengthCm'])

hist1.set_title('Distribution and Density of Sepal Length for Iris-setosa')
hist2 = sns.distplot(iris_data1['SepalWidthCm'])

hist2.set_title('Distribution and Density of Sepal Width for Iris-setosa')
hist3 = sns.distplot(iris_data1['PetalLengthCm'])

hist3.set_title('Distribution and Density of Petal Length for Iris-setosa')
hist4 = sns.distplot(iris_data1['PetalWidthCm'])

hist4.set_title('Distribution and Density of Petal Width for Iris-setosa')
# Plotting the boxplot for Iris Species



subset1 = iris_data1[[1,2,3,4]]

subset1.head()



v1 = sns.violinplot(subset1)
# Iris Versicolor



iris_data2 = iris_data[iris_data['Species'] == 'Iris-versicolor']

iris_data2.head()
# Getting the Distributions for Iris-Versicolor



hist5 = sns.distplot(iris_data2['SepalLengthCm'])

hist5.set_title('Distribution and Density of Sepal Length for Iris-Versicolor')
hist6 = sns.distplot(iris_data2['SepalWidthCm'])

hist6.set_title('Distribution and Density of Sepal Width for Iris-Versicolor')
hist7 = sns.distplot(iris_data2['PetalLengthCm'])

hist7.set_title('Distribution and Density of Petal Length for Iris-Versicolor')
hist8 = sns.distplot(iris_data2['PetalWidthCm'])

hist8.set_title('Distribution and Density of Petal Width for Iris-Versicolor')
# Plotting the boxplot for Iris Versicolor



subset2 = iris_data2[[1,2,3,4]]

subset2.head()



v2 = sns.violinplot(subset2)
# Iris Virginica



iris_data3 = iris_data[iris_data['Species'] == 'Iris-virginica']

iris_data3.head()
hist9 = sns.distplot(iris_data2['SepalLengthCm'])

hist9.set_title('Distribution and Density of Sepal Length for Iris-Virginica')
hist10 = sns.distplot(iris_data2['SepalWidthCm'])

hist10.set_title('Distribution and Density of Sepal Width for Iris-Virginica')
hist11 = sns.distplot(iris_data2['PetalLengthCm'])

hist11.set_title('Distribution and Density of Petal Width for Iris-Virginica')
hist12 = sns.distplot(iris_data2['PetalWidthCm'])

hist12.set_title('Distribution and Density of Petal Width for Iris-Virginica')
# Boxplot for Iris Virginica



subset3 = iris_data3[[1,2,3,4]]

subset3.head()



v3 = sns.violinplot(subset3)
## Plotting the pairplot for the data



sns.pairplot(iris_data,hue = 'Species')