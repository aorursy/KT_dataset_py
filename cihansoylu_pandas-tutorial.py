#This notebook is a tutorial on the basic functionality of the pandas library

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import datasets # example datasets in sklearn library
#Before getting into pandas functionality we will load an example dataset from sklearn library 
iris = datasets.load_iris() #load iris dataset
type(iris) #iris is a bunch instance which is inherited from dictionary
iris.keys() #keys of the dictionary
#The description of the dataset. We see that the dataset has 4 features and a 3 classes.
#There are 150 examples in the dataset
print(iris.DESCR) 
#The feature values are stored under the key 'data' and it is an np.array
type(iris.data)
#each row of this array correspond to an example data point and each column corresponds to a feature
print(iris.data)
#The feature name corresponding to each column is stored in 'feature_names'
print(iris.feature_names)
#The classes of each data point is stored in the 'target' key
print(iris.target)
#The name of the classes corresponding to 0,1,2 can be found in
print(iris.target_names)
#Now we are ready to create a pandas dataframe using the iris dataset
#We will do this in two different ways:
# 1-Creating a dataframe from an np.array
# 2-Creating a dataframe from a .csv file

# 1- To create a dataframe from an np.array simply call pd.DataFrame(array) 
iris_dataframe = pd.DataFrame(iris.data)
iris_dataframe.head() #first five data points
#As you see above the feature names are set as 0,1,2,3. We can change them as follows
iris_dataframe.columns = iris.feature_names
# we could also do this when we are creating the data frame as follows
#iris_dataframe = pd.DataFrame(iris.data, columns = iris.feature_names)
iris_dataframe.head()
# Now we want to add the classes of each data point as a new column:
iris_dataframe['target'] = iris.target
iris_dataframe.head()
# You can get a column of a dataframe by using the dot notation
iris_dataframe['sepal length (cm)']
#Another way to create a data frame is reading from a .csv file. For this we first export the
#dataframe we created to a .csv file and then create a new data frame by reading the same file.
#We will use to_csv function. Setting the index and header parameters to False prevent the 
#index and header written on file.
iris_dataframe.to_csv('iris.csv', index=False, header = False)
ls
#Now we will create a new data frame using read_csv() function. We will pass the path of the file
#to this function together with the names parameter. See the documentation for details.
columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
iris_dataframe_new = pd.read_csv('iris.csv', names = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class'])
iris_dataframe_new.head()
iris_dataframe_new.index
iris_dataframe_new.columns
#you can sort the data according to a particular column
iris_dataframe_new.sort_values(by = 'sepal width')
#selecting certain rows
iris_dataframe_new[1:10]
#selection by labels
iris_dataframe_new.loc[1:10, ['sepal length', 'petal length']]
#selection by locations
iris_dataframe_new.iloc[1:10, [0,2]]
#masking by a single column values
iris_dataframe_new[iris_dataframe_new['sepal length'] > 7]
#filtering the dataframe using isin function 
iris_dataframe_new[iris_dataframe_new['sepal length'].isin([7.2])]
#applying functions
iris_dataframe_new['sepal length'].apply(lambda x : x**2).head()
iris_dataframe_new.apply(lambda x : x**2).head()
