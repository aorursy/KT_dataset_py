# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#for data visualization

import seaborn as sns

import matplotlib.pyplot as plt 



from scipy import stats



#load the dataset



iris = pd.read_csv("/kaggle/input/iris/Iris.csv") #load the dataset

# find dimensions of the dataset

iris.shape  #150 rows and 6 columns
# get the column names

columns = list(iris)

iris.columns

#Dislpay a sample of the dataset



iris.head()
# Get the count for each class

iris['Species'].value_counts() #50 each for each class . Total 3 classes.
# Method 1 : Pandas.DataFrame.info(self, verbose=None, buf=None, max_cols=None, memory_usage=None, null_counts=None)

# This method prints information about a DataFrame including the index dtype and column dtypes, non-null values and memory usage.

# Returns None.The memory_usage parameter allows deep introspection mode, specially useful for big DataFrames and fine-tune memory optimization:



iris.info(memory_usage='deep') #No missing values found.
#dropping the Id column as it is unecessary, axis=1 specifies that it should be column wise, inplace =1 means the changes should be reflected into the dataframe

iris.drop('Id',axis=1,inplace=True) 
# Method 2 : Pandas.DataFrame.describe(self, percentiles=None, include=None, exclude=None)

# Generate descriptive statistics that summarize the central tendency, dispersion and 

# shape of a dataset’s distribution, excluding NaN values.



iris.describe()
fig, ax =plt.subplots(1,4,figsize=(20,6))



    

sns.distplot(iris.SepalLengthCm,ax=ax[0])

sns.distplot(iris.SepalWidthCm,ax=ax[1])

sns.distplot(iris.PetalLengthCm,ax=ax[2])

sns.distplot(iris.PetalWidthCm,ax=ax[3])



fig.show()

sns.distplot(iris.SepalLengthCm, kde=True, fit=stats.norm);
sns.boxplot(x="Species", y="SepalLengthCm", data=iris)
sns.violinplot(x="Species",y="SepalLengthCm",data = iris)

plt.title("Violin plot for SepalLengthCm and Species")

plt.show()

ax = sns.swarmplot(x="SepalLengthCm", y="Species", data=iris)
g = sns.FacetGrid(iris, col="Species")

g.map(plt.hist, "SepalLengthCm");
sns.FacetGrid(data=iris,hue="Species",height=5).map(plt.scatter,"SepalLengthCm","SepalWidthCm").add_legend()
with sns.axes_style("white"):

    sns.jointplot(x="SepalLengthCm", y="SepalWidthCm", kind="hex", color="k",data=iris);
sns.jointplot(x="SepalLengthCm", y="SepalWidthCm", data=iris, kind="kde");

plt.figure(figsize=(8,6)) 

sns.heatmap(iris.corr(),annot=True,cmap="YlGnBu") #draws  heatmap with input as the correlation matrix calculted by(iris.corr())

plt.show()  
g = sns.pairplot(iris, hue="Species", palette="husl")