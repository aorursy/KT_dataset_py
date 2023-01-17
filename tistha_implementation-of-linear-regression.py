import pandas as pd

# above library is use to read the dataset



import matplotlib.pyplot as plt

%matplotlib inline

# above library is use to plot the graph to give pictorial representation of the datset
dataset = pd.read_csv('../input/worldstatscsv/worldstats.csv')

# Here read_csv method is used because it is reading data from a csv file.



dataset.head()

# .head() function is use to display first 5 values of the dataset. If any number is mentioned in the parenthesis then

# those many values are displayed
dataset.country.value_counts() # value_counts returns the count of occurences of each of the unique values in the column
dataset.describe()



# The .describe() method is use to give a descriptive exploration on the dataset
dataset.info()



# This function is used to get a concise summary of the dataframe.
dataset.shape



# Returns the number of rows and columns
dataset.dtypes



# Returns the datatype of the values in each column in the dataset
dataset.isnull().sum()



# .isnull() is used to check for null values in the dataset. It returns result in true/false manner.



# .sum() used with isnull() gives the combined number of null values in the dataset if any.
dataset.hist(grid=True, figsize=(20,10), color='purple')
plt.figure(figsize=(20,10))

sns.pairplot(dataset, diag_kind='kde')
import seaborn as sns

# the above library is used to draw the graph



sns.set(color_codes=True)

# this code is used to display the background grid i.e. the rectangular boxes in the graph
X = dataset[['year', 'Population']]

sns.regplot(x="year", y="Population", data=X)



plt.xlabel('YEAR')  # Provides the name to the x-axis

plt.ylabel('POPULATION')    # Provides the name to the y-axis

plt.show()  # Used to display the graph

Y = dataset[['GDP', 'Population']]

sns.regplot(x="GDP", y="Population", data=Y)

Z = dataset[['GDP', 'year']]

sns.regplot(x="GDP", y="year", data=Z)

from sklearn.linear_model import LinearRegression  

# This library is used for loading the linear model for implementing Linear Regression



reg = LinearRegression(normalize=True)

# It loads the LinearRegression model into the variable reg that will be used to train and test the model
ds1 = dataset[['year']] # independent variable

ds2 = dataset.GDP # dependent variable

      

# fits the linear model i.e. trains the model using the uploaded dataset
reg.fit(ds1, ds2)
reg.fit(ds1,ds2)
reg.predict([[1997]])
dss1 = dataset[['year', 'Population']] # independent variable

dss2 = dataset.GDP # dependent variable

      
reg.fit(dss1, dss2)
reg.predict([[2020, 234255034]])
reg.score(dss1, dss2)