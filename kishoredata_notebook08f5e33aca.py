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





import matplotlib.pyplot as plt

iris_data=pd.read_csv("../input/Iris.csv")
# cleaning data

#iris_data.info()

# find the colimns name

iris_data.columns
# checking the iris is null or nan value

iris_data.isnull().sum()
# find the value counts for Species

iris_data['Species'].value_counts()
# A seaborn jointplot shows bivariate scatterplots and univariate histograms in the same figure

import seaborn as sns

sns.jointplot(x="SepalLengthCm", y="SepalWidthCm", data=iris_data, size=5)
# BOX plot

plt.boxplot(iris_data['SepalWidthCm'], 1)
sns.boxplot(x="Species", y="PetalLengthCm", data=iris_data)
# scatter plot

plt.scatter(iris_data['PetalLengthCm'],iris_data['SepalWidthCm'])
sns.regplot(iris_data['PetalLengthCm'],iris_data['SepalWidthCm'],color='r')