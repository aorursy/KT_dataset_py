import pandas as pd
import numpy as np
data = {'weekday': ['Sun','Sun','Mon','Tue'],
        'city': ['Austin','Dallas','Pittsburgh','California'],
        'vistors': [145,135,676,456]
       }
users = pd.DataFrame(data)
print(users)
# Read dataset
titanic_data = pd.read_csv("../input/train.csv")
# print type of the object
type(titanic_data)
# Shape of the dataframe
titanic_data.shape
# show the columns in the dataframe
titanic_data.columns
# show first n records
titanic_data.head(2)
# show last n records
titanic_data.tail(2)
# Change column names of the dataframe
# titanic_data.columns = ['a', 'b']
# titanic_data.rename(columns={'Name': 'PName', 'Ticket': 'Number'}, inplace=True)
# show the data types and missing values in the dataset
titanic_data.info()
# Summary statistics of the dataframe - For only numerical variables (Integer and float)
titanic_data.describe()
# Summary stats for categorical variables
titanic_data.describe(include = ['object','bool'])
# Convert datatype of a column in the dataframe
titanic_data['Pclass'] = titanic_data['Pclass'].astype('object')
titanic_data.dtypes
# Indexing using square barckets
titanic_data['Name'][0]
#titanic_data.Name[0]
# Select only few columns in the data
titanic_data[['Survived', 'Pclass', 'Name', 'Sex']].head()
# Drop few columns in the data
titanic_data.drop(['Ticket','Cabin'],axis = 1).head()
# slicing the df using loc (loc is used with column  names)
titanic_data.loc[0,'Fare']
# slicing all the rows but few columns in df
titanic_data.loc[:,'Survived':'Sex'].head(2)
# Slicing few rows but all columns in the df
titanic_data.loc[0:4,:].head(2)
# Slicing selected rows and columns
titanic_data.loc[0:3,['Sex','Age']]
# Slicing the df using iloc (iloc is used with index numbers)
titanic_data.iloc[0,9]
# Slicing the data frame rows by iloc
titanic_data.iloc[5:8,:]
# Slicing selected rows and columns using iloc
titanic_data.iloc[[0,4,6], 0:2]
# Slicing the last n rows of the dataframe using iloc
titanic_data.iloc[-5:,:]
# Filter only male records from df
titanic_data[titanic_data['Sex'] == 'male'].shape
# Filtering with a boolean series
titanic_data[titanic_data.Age > 50].shape
# Filter by a string pattern in a column
titanic_data[titanic_data['Name'].str.contains('Sir')]
# Multiple conditions within a filter
titanic_data[(titanic_data.Age >= 50) & (titanic_data.Fare > 30)].shape
# Filter on multiple conditions within same column
titanic_data[titanic_data['Embarked'].isin(['C','Q'])].shape
# Drop any rows with missing values
titanic_data.dropna(how = 'any').shape
# Group by a column (Avg survival rate by gender)
titanic_data.groupby("Sex")["Survived"].mean()
# Group by jointly on two columns (Avg survival rate by gender and Pclass)
titanic_data.groupby(["Sex","Pclass"])["Survived"].mean()
# In the Pandas version, the grouped-on columns are pushed into the MultiIndex of the resulting Series by defaul
# More closely emulate the SQL result and push the grouped-on columns back into columns in the result, 
# you an use as_index=False
titanic_data.groupby(["Sex","Pclass"],as_index = False)["Survived"].mean()
# Average fare by passenger class and sort it by descending order
Fare_by_class = titanic_data.groupby("Pclass",as_index = False)["Fare"].mean()
Fare_by_class.sort_values(['Fare'],ascending = False).head()
