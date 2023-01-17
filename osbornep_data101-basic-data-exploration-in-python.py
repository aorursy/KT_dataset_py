import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
# LOAD DATA FROM LOCAL DIRECTORY

#iris_data = pd.read_csv(r'C:\Users\Phil\Sync\Data Science MSc\Data\IRIS.csv')



# LOAD DATA FOR KAGGLE NOTEBOOK

iris_data = pd.read_csv('../input/iris/Iris.csv')
# Top 5 rows

iris_data.head()
# Top k rows

iris_data.head(2)
# Last 5 rows

iris_data.tail()
# 1. ID column not in original source so we first select only those that match with the original source

iris_data = iris_data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm','Species']]

# 2. The names of each column are slightly different so we re-name these to match the original source

iris_data.columns = ['sepal_length', 'sepal_width', 'petal_length','petal_width','species']
iris_data.head()
iris_data.dtypes
# Top 5 rows of 'sepal length' column

iris_data['sepal_length'].head()
# Top 5 rows of 'sepal length' column

iris_data.sepal_length.head()
# Top 5 rows of 'sepal length' column

iris_data.sepal_length.head()
# First row

iris_data.iloc[0]
# Rows between 3rd and 7th 

iris_data.iloc[2:6]
# First row and column element

iris_data.iloc[0,0]
# All rows and columns - top 5 rows only shown

iris_data.iloc[:,:].head()
# Rows between 3rd and 7th and all columns

iris_data.iloc[2:6,:]
# All rows and columns between 3rd and 5th - top 5 rows only shown

iris_data.iloc[:,2:5].head()
# 1st and 2nd row and 4th and 5th column

iris_data.iloc[[0,1],[3,4]]
# Rows between the 3rd and 7th and the 4th and 5th column

iris_data.iloc[2:6,[3,4]]
# 1st and 2nd rows and the 3rd and 5th

iris_data.iloc[[0,1],2:5]
# Rows between the 3rd and 7th and columns between the 3rd and 5th

iris_data.iloc[0:5,2:5]
iris_data['species'].unique()
# Select Iris-setosa data - top 5 rows only shown

iris_data[iris_data['species']=='Iris-setosa'].head()
# Select Iris-versicolor data - top 5 rows only shown

iris_data[iris_data['species']=='Iris-versicolor'].head()
# Select data with sepal_length EQUAL TO 4.9

iris_data[iris_data['sepal_length']==4.9]
# Select data with sepal_length LESS THAN 4.5

iris_data[iris_data['sepal_length']<4.5]
# Select data with sepal_length LESS THAN OR EQUAL TO 4.5

iris_data[iris_data['sepal_length']<=4.5]
# Select data with sepal_length GREATER THAN 7.4

iris_data[iris_data['sepal_length']>7.4]
# Select data with sepal_length GREATER THAN OR EQUAL TO 7.5

iris_data[iris_data['sepal_length']>=7.4]
# Select Iris-setosa OR Iris-versicolor data - top 5 rows only shown

iris_data[(iris_data['species']=='Iris-setosa') | (iris_data['species']=='Iris-versicolor')].head()
# Select Iris-setosa AND Iris-versicolor data - no rows have both so none are returned

iris_data[(iris_data['species']=='Iris-setosa') & (iris_data['species']=='Iris-versicolor')].head()
# Select Iris-setosa AND Sepal Length EQUAL TO 4.9 

iris_data[(iris_data['species']=='Iris-versicolor') & (iris_data['sepal_length']==4.9)]
# Select Sepal Length EQUAL TO 4.8 OR Sepal Length EQUAL TO 4.9 

iris_data[(iris_data['sepal_length']==4.8) | (iris_data['sepal_length']==4.9)]
# Select Sepal Length EQUAL TO 4.8 OR Sepal Width EQUAL TO 3.0

iris_data[(iris_data['sepal_length']==4.8) & (iris_data['sepal_width']==3.0)]
# Select [Sepal Length EQUAL TO 4.8 OR Sepal Length EQUAL TO 4.9] AND Iris-setosa

iris_data[((iris_data['sepal_length']==4.8) | (iris_data['sepal_length']==4.9)) & (iris_data['species']=='Iris-setosa')]
# Select Iris-setosa data - Save to new variable 'iris_data_setosa'

iris_data_setosa = iris_data[iris_data['species']=='Iris-setosa']

iris_data_setosa.head()
iris_data.describe()
iris_data['species'].unique()
list(iris_data)
# Basic Matplotlib plot with manually defined data

plt.plot([1, 2, 3, 4], [1, 4, 9, 16])



# Visual elements

plt.title('Basic Plot Example')

plt.ylabel('y-label')

plt.xlabel('x-label')



plt.show()
# Basic Matplotlib plot with manually defined data

# Vary the linestyle

plt.plot([1, 2, 3, 4], [1, 4, 9, 16], '--', linewidth=5)



# Visual elements

plt.title('Basic Plot Example')

plt.ylabel('y-label')

plt.xlabel('x-label')



plt.show()
# Basic Matplotlib plot with manually defined data

# Vary the colour

plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'r--')



# Visual elements

plt.title('Basic Plot Example')

plt.ylabel('y-label')

plt.xlabel('x-label')



plt.show()
# Basic Matplotlib plot with manually defined data

# Vary the marker type

plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'r.')



# Visual elements

plt.title('Basic Plot Example')

plt.ylabel('y-label')

plt.xlabel('x-label')



plt.show()
# Basic Matplotlib plot with manually defined data

# Vary the marker type

plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'r.', markersize = 20)



# Visual elements

plt.title('Basic Plot Example')

plt.ylabel('y-label')

plt.xlabel('x-label')



plt.show()
# Remind ourselves of the data with the .head() preview

iris_data.head()
# Basic Matplotlib plot for the Iris data

plt.plot(iris_data['sepal_length'], iris_data['sepal_width'], 'b.', markersize = 10)



# Visual elements

plt.title('Sepal Length vs Sepal Width')

plt.xlabel('Sepal Length')

plt.ylabel('Sepal Width')



plt.show()
# Basic Matplotlib plot for the Iris data

plt.plot(iris_data['petal_length'], iris_data['petal_width'], 'b.', markersize = 10)



# Visual elements

plt.title('Petal Length vs Petal Width')

plt.xlabel('Petal Length')

plt.ylabel('Petal Width')



plt.show()
import seaborn as sns
# Basic Seaborn scatter plot with markers coloured by the 'Species' feature

ax = sns.scatterplot(x="sepal_length", y="sepal_width", hue="species", data=iris_data)



# Visual elements

plt.title('Sepal Length vs Sepal Width')

plt.xlabel('Sepal Length')

plt.ylabel('Sepal Width')



plt.show()
sns.set_style("white")
# Set figure size

plt.figure(figsize=(12,8))



# Basic Seaborn scatter plot with markers coloured by the 'Species' feature

ax = sns.scatterplot(x="sepal_length", y="sepal_width", hue="species", data=iris_data, s=50)



# Visual elements

plt.title('Sepal Length vs Sepal Width')

plt.xlabel('Sepal Length')

plt.ylabel('Sepal Width')



plt.show()