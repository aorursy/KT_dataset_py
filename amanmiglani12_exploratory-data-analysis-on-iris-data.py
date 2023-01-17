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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns 
df_iris = pd.read_csv('../input/iris/Iris.csv')
# Q). How many data-point(no. of rows) and features(no. of columns) are there?

df_iris.shape
# Q). What are the columns or feature names in dataset ?

df_iris.columns
# Q). How many data points or flower for each class/spieces are present ?

df_iris['Species'].value_counts()

# balanced vs imbalanced dataset 

# Iris is a balanced dataset since the number of data points for each speicies are equal(50)

# What is imbalanced dataset ? 

# Ans.) Imagine if we have dataset in which there are only 2 spieces and now if one spieces has 900 datapoints  

#        and other has 100 only then such a data set is called imbalanced data set. 

#        for example: dataset from hospital having species cancer and non-cancer as 

#        we know most of datapoint will be non-cancer patient, so this is imbalanced dataset
## 2-D Scatter-Plot :

df_iris.plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm')



# Key point: ALways see the labels and scale of graph



# cannot make much sense out it after plotting : we can only see that  4<sepal_length<8 and 2<Sepalwidth<4.5

# Now we will do colorthe points by their class-labesls/flower-type
# 2-D scatter plot with color for each flower type/class

sns.set_style("whitegrid")

sns.FacetGrid(df_iris,hue='Species',size=4).map(plt.scatter,'SepalLengthCm','SepalWidthCm').add_legend()

plt.show()



# Notice Blue point can be easily seperated from red and blue data points by drwaing a line

# But green and orange  data points are not easily seperable 



# pairwise scatter plot : Pair-Plot

# One disadvanatge of Pair Plot : Cant' be Used when number of features are high.

sns.set_style("whitegrid")

sns.pairplot(df_iris,hue='Species',size=3)
# 1-D scatter plot using just one Feature 

# 1-D scatter plot of Petal-Length

setosa=df_iris[df_iris['Species']=='Iris-setosa']

virginca= df_iris[df_iris['Species']=='Iris-virginica']

versicolor=df_iris[df_iris['Species']=='Iris-versicolor']

plt.plot(versicolor['PetalLengthCm'],np.zeros_like(versicolor['PetalLengthCm']),'o') # it will create a plot such that x-axis is PetalLengthCm and y-axis valuesare zeros

plt.plot(setosa['PetalLengthCm'],np.zeros_like(setosa['PetalLengthCm']),'ro')

plt.plot(virginca['PetalLengthCm'],np.zeros_like(versicolor['PetalLengthCm']),'go')







# disadvantage of  1-D scatter plot are very hard to read as point are overlapping a lot.
# Histogram a better way of visualizing 1-D scatter plots bcz we can tell about the no. of data points present between 2 points



sns.FacetGrid(df_iris,hue='Species',size=5).map(sns.distplot,'PetalLengthCm').add_legend()





#here the Smooth curve is smooth curve of histogram using kde(kernel density estimator) and is called P.D.F(probability density fucntion)
sns.FacetGrid(df_iris,hue='Species',size=5).map(sns.distplot,'PetalWidthCm').add_legend()
sns.FacetGrid(df_iris,hue='Species',size=5).map(sns.distplot,'SepalLengthCm').add_legend()
sns.FacetGrid(df_iris,hue='Species',size=5).map(sns.distplot,'SepalWidthCm').add_legend()
#Plot CDF of petal_length



counts, bin_edges = np.histogram(setosa['PetalLengthCm'], bins=10 ,density = True)

pdf = counts/(sum(counts))

print(pdf);

print(bin_edges)



#compute CDF

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf)

plt.plot(bin_edges[1:], cdf)

print("1st setosa")

counts, bin_edges = np.histogram(setosa['PetalLengthCm'], bins=10 ,density = True)

pdf = counts/(sum(counts))

print(pdf);

print(bin_edges)



#compute CDF

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf)

plt.plot(bin_edges[1:], cdf)







print("2nd versicolor")

counts, bin_edges = np.histogram(versicolor['PetalLengthCm'], bins=10 ,density = True)

pdf = counts/(sum(counts))

print(pdf);

print(bin_edges)



#compute CDF

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf)

plt.plot(bin_edges[1:], cdf)







print("3rd.Virginica")

counts, bin_edges = np.histogram(virginca['PetalLengthCm'], bins=10 ,density = True)

pdf = counts/(sum(counts))

print(pdf);

print(bin_edges)



#compute CDF

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf)

plt.plot(bin_edges[1:], cdf)



# Box-plot with Whiskers : another simple method of visualizing the 1-D scatterplot more intuitively

# It uses the concept of mean , median ,Percentile and Quantile

sns.boxplot(x='Species',y='PetalLengthCm',data=df_iris)



# By seeing the plot we can get the 25th ,50th ,75th percentiles.  
#Violinplot is the combination of the histogram with pdf and Box-Plot

#Denser region of the data are fatter and Sparser ones are thinner in violin plot



sns.violinplot(x='Species',y='PetalLengthCm',data=df_iris)