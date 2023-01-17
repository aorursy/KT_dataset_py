# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# 1. step = Load data and check the structure of the data frame
iris_data = pd.read_csv('../input/Iris.csv')

iris_data.head(5)
# 2. step = Check the data types of the columns

# 2.a I can check each column separately with dtype
#iris_data['Id'].dtype # int64
#iris_data['SepalLengthCm'].dtype #float64
#iris_data['SepalWidthCm'].dtype #float64
#iris_data['PetalLengthCm'].dtype #float64
#iris_data['PetalWidthCm'].dtype #float64
#iris_data['Species'].dtype # O

# 2.b or I can check data types of all columns with one line of code with dtypes = better
iris_data.dtypes
# 3. Do we have any data missing?

missing = iris_data.isnull().sum()
missing

# No values are missing, data set looks pretty cleared
# 4. Let's plot to see how the data look

# 4.a Histogram
import matplotlib.pyplot as plt
import seaborn as sns # let's use nicer visualization package

sns.set() # the point here is to set the seaborn as the default visuals, however we continue to plot the contents of the histogram using matplotlib
_=plt.hist(iris_data['PetalLengthCm'])
_=plt.xlabel('Petal length')
_=plt.ylabel('Number of observations')
plt.show()
# 4.b in Histogram we can play with the size of the bins, let's try to see how it would change our graph
sns.set()
_=plt.hist(iris_data['PetalLengthCm'], bins=20)
_=plt.xlabel('Petal length')
_=plt.ylabel('Number of observations')
plt.show()

# with twice as many bins we can see that the distribution of the variable is rather spread out giving us a rather realistic look over the data. 
# 5. Bee swarm plot
# sns.set() does not need to be used as we are using a plot from seaborn directly
_=sns.swarmplot(x='Species', y='PetalLengthCm', data=iris_data)
_=plt.xlabel('Species') # labels can be added using matplotlib
_=plt.ylabel('Petal length')
plt.show()

# The petal length differ across species.
# Setosa has the smallest variation of petal length. Virginica has the biggest variation of petal length.
# Another interesting point is the difference between petal length between the different Species. We can see that Setosa species has petals distinctively shorter that the other two species. This is also visible on the histogram above.
# 6. Empirical cumulative distribution function (ECDF)
# x = is the value we want to measure (petal length in our case)
# y = is the fraction of data points that have the measurement smaller than x value

# 6.1 Compute the ECDF
# create a function that will return x, y just as we need it for plotting the ECDF = general function

def ecdf(data):
    # number of data points: n
    n = len(data)
    # x-data for the ECDF: x
    x=np.sort(data)
    # y-data for the ECDF: y
    y=np.arange(1, n+1)/n
    return x,y

# 6.2 Compute the ECDF for the versicolor data (only one species from the data set)
versicolor=iris_data[iris_data.Species=='Iris-versicolor']
versicolor_petal_length = versicolor.PetalLengthCm
versicolor_petal_length

x_vers, y_vers = ecdf(versicolor_petal_length)

# Generate plot
_ = plt.plot(x_vers, y_vers, marker='.', linestyle='none')

# Label the axes
plt.xlabel('Versicolor')
plt.ylabel('ECDF')

# Display the plot
plt.show()
# 6.3 repeat data manipulation for the rest of the Species
# Setosa
setosa=iris_data[iris_data.Species=='Iris-setosa']
setosa_petal_length = setosa.PetalLengthCm
setosa_petal_length

# Virginica
virginica=iris_data[iris_data.Species=='Iris-virginica']
virginica_petal_length = virginica.PetalLengthCm
virginica_petal_length

# 6.4 Compute ECDFs for all three species and plot a comparative graph
x_set, y_set = ecdf(setosa_petal_length)
x_vir, y_vir = ecdf(virginica_petal_length)
x_vers, y_vers = ecdf(versicolor_petal_length)

_=plt.plot(x_set, y_set, marker='.', linestyle='none')
_=plt.plot(x_vers, y_vers, marker='.', linestyle='none')
_=plt.plot(x_vir, y_vir, marker='.', linestyle='none')

# Add a legend
plt.legend(('setosa', 'versicolor', 'virginica'), loc='lower right')
_ = plt.xlabel('petal length (cm)')
_ = plt.ylabel('ECDF')