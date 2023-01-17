# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
iris_data = pd.read_csv("../input/Iris.csv") # Load the dataset
iris_data.head(3) 
iris_data.info() # Overview of the data; There are no missing values
print(iris_data.columns.tolist()) # Look at the columns
iris_data.drop('Id', axis = 1, inplace = True) # Removing the ID column - inplace ensures the column is removed from the dataframe
print(iris_data.columns.tolist()) # Look at the columns to confirm ID is removed
iris_data.hist()
fig=plt.gcf()
fig.set_size_inches(18,16)
plt.show()
plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.violinplot(x='Species',y='PetalLengthCm',data=iris_data)
plt.subplot(2,2,2)
sns.violinplot(x='Species',y='PetalWidthCm',data=iris_data)
plt.subplot(2,2,3)
sns.violinplot(x='Species',y='SepalLengthCm',data=iris_data)
plt.subplot(2,2,4)
sns.violinplot(x='Species',y='SepalWidthCm',data=iris_data)
iris_split = [iris_data['Species'].value_counts()]
plt.pie(iris_split[0], autopct='%1.1f%%', colors = ['orange', '#3498db', 'green'], startangle=90)
plt.title('Iris Species Split', size = 12)
#plt.legend(labels = ['Non-Legendary', 'Legendary', 'abc'], loc = 'best')
plt.show()
fig = iris_data[iris_data.Species == 'Iris-setosa'].plot.scatter(x = 'PetalLengthCm', y = 'PetalWidthCm', color = 'orange', label = 'Setosa')
iris_data[iris_data.Species == 'Iris-versicolor'].plot.scatter(x = 'PetalLengthCm', y = 'PetalWidthCm', color = 'blue', label = 'versicolor', ax = fig)
iris_data[iris_data.Species == 'Iris-virginica'].plot.scatter(x = 'PetalLengthCm', y = 'PetalWidthCm', color = 'green', label = 'virginica', ax = fig)
fig.set_xlabel("Petal Length")
fig.set_ylabel("Petal Width")
fig.set_title("PETAL Length VS Width")
fig = plt.gcf()
fig.set_size_inches(12,8)
plt.show()
fig = iris_data[iris_data.Species == 'Iris-setosa'].plot.scatter(x = 'SepalLengthCm', y = 'SepalWidthCm', color = 'orange', label = 'Setosa')
iris_data[iris_data.Species == 'Iris-versicolor'].plot.scatter(x = 'SepalLengthCm', y = 'SepalWidthCm', color = 'blue', label = 'versicolor', ax = fig)
iris_data[iris_data.Species == 'Iris-virginica'].plot.scatter(x = 'SepalLengthCm', y = 'SepalWidthCm', color = 'green', label = 'virginica', ax = fig)
fig.set_xlabel("Sepal Length")
fig.set_ylabel("Sepal Width")
fig.set_title("SEPAL Length VS Width")
fig = plt.gcf()
fig.set_size_inches(12,8)
plt.show()