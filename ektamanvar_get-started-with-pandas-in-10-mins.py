import pandas as pd

#import dataset
data = pd.read_csv('/kaggle/input/titanic/train.csv')

#Export dataset
data.to_csv('output.csv')

#Returns by default top five rows
data.head()
#Returns by default last five rows
data.tail()
#Returns column names
data.columns 
#Returns column data types
data.dtypes 
#Returns (Rows, Columns) of th dataset
data.shape 
#Returns total count of non NA values for each column
data.count()
#Returns basic statistical information of numeric columns only
data.describe() 
#Returns information of string columns
data.describe(include=object)
#Returns information about dataframe like index dtype and column dtypes, non-null values and memory usage
data.info()
#Returns distinct count of observations for each column
data.nunique()
#Returns count of unique values for series(columns in our case)
data['Pclass'].value_counts()
#Rename the column names
data.rename(columns= {"Name":"Full Name"})
#Drop columns
data.drop(columns=['Name','Survived','Pclass'])
#Selects one column data
data['Name']
#Selects more than one column data
data[['Sex','Name']]
#Filters data based on condition
data[data['Age']>50]
#Filters all rows and 3 columns(sex,pclass,age)
data.loc[:, ['Sex','Pclass','Age']]
#Filters 100 to 400 rows and 2 columns(survived, sex)
data.loc[100:400 :,['Survived','Sex']]
#Filters all rows and columns from survived to sex (Survived, Pclass, Name, Sex)
data.loc[:, 'Survived':'Sex']
#Filters rows based on condition (sex=female) and all columns
data.loc[data['Sex']=='female', ]
#Filters all rows and 1 to 4 columns
#index starts from zero and ignores last index while filtering  
data.iloc[ :,1:4]
#Filters all rows and 1,4,6 columns
data.iloc[:,[1,4,6]]
#Filters rows from 150 to 400 and and 2,7,3 columns
data.iloc[150:400:,[2,7,3]]
#Returns Survived column wise count
data.groupby('Survived').count()
#Returns max age based on sex column
data.groupby('Sex')['Age'].max()
#Returns min age based on sex, survived columns
data.groupby(['Sex','Survived'])['Age'].min()
#Multiple aggregation functions  
#Returns min,max age based on parch & survived
data.groupby(['Parch','Survived'])['Age'].agg([min, max])
#Sorts the dataset based on specified single column(default it sorts in ascending order)
data.sort_values(by='Name')
#Sorts the dataset based on specified single column in decending order
data.sort_values(by='Name', ascending=False)
#Returns True or False 
data.isnull()
data.isna()
#Fills all NA's with zero for both string nd numeric

#NA's can be filled with mean,median or mode as well. 
data.fillna(0)
#Drops rows if row have at least one NA value
data.dropna(how='any') 
#Drops rows if the row have all NA values
data.dropna(how='all')
#Drops columns if the column have at least one NA value
data.dropna(axis='columns', how='any')
#Checks if whole row appears elsewhere with same values. 
data.duplicated()
#checks if there is any duplicate values of particular column
data.duplicated('Age')
#drops duplicate records 
data.drop_duplicates()
#drops duplicates from particular column
data.drop_duplicates('Age')
#sets index based on specified column 
data.set_index('Sex')
#sets index based on specified columns
data1 = data.set_index(['Sex','PassengerId'])
data1
#reset index 
data1.reset_index()
#melt() function is used to convert dataframe from wide format to long format
data.melt(id_vars='PassengerId')
#we can use melt() function for particular columns as well
data.melt(id_vars='PassengerId', value_vars=['Survived','Sex'], var_name='Columns', value_name='Column_values')
#pivot() function is used to reshape dataframe based on index/columns values. Results into multiindex 
# don't support aggregation function

data.pivot(index='PassengerId', columns='Sex')
#we can use pivot() function for particular columns as well
data.pivot(index='PassengerId', columns='Sex', values=['Survived','Pclass','Age'])


#pivot_table is used for data aggregation 
#Obersvations are filled with sum of age values

data.pivot_table(index='PassengerId', columns='Sex', values='Age', aggfunc='sum')

#Let's create data frames to demonstrate Merge and Concat functions

df1 = pd.DataFrame({"x1": ["a","b","c","d"], "x2":[12.0, 23.2, 56, 45.4]})
df2 = pd.DataFrame({"x1": ["a","b","c","e"], "x3":[9.5, 37.0, 77,38.9]})
df1
df2
#Merge function 
#merges come data based on x1 column: inner join (a,b,c)
pd.merge(df1,df2,on="x1")
#merges both data based on x1 column: outer join (a,b,c,d,e)
#while merging if data is not there then it will replace with NaN value

pd.merge(df1,df2,on="x1", how="outer")
#merges common data from both dataset and remaining data from left dataset 
#while merging if data is not there then it will replace with NaN value

pd.merge(df1,df2,on="x1", how="left")
#merges common data from both dataset and remaining data from right dataset 
#while merging if data is not there then it will replace with NaN value

pd.merge(df1,df2,on="x1", how="right")
#concat function
#by default performs outer join and works row wise
pd.concat([df1,df2])
#axis will be labeled 0, …, n - 1
pd.concat([df1,df2], ignore_index=True)
#concatnates column wise
pd.concat([df1,df2], axis=1)
pd.concat([df1,df2], join="inner")
#Load the dataset, for date formating I am using applestock price dataset.

data1 = pd.read_csv('https://raw.githubusercontent.com/Ekta-Manvar/Pandas-For-Data-Analysis/master/applestock.csv')

data1.head()
#check the datatypes
data1.dtypes

#date column is object type not datetime format
#to_datetime() - converts any format to datetime format

data1['Date'] = pd.to_datetime(data1['Date'])
data1.dtypes
#extract year, month, day from date column 

data1['Year'] = data1['Date'].dt.year
data1['Month'] = data1['Date'].dt.month
data1['day'] = data1['Date'].dt.day

data1.head()
#pd.DatetimeIndex() - sets date as index
data1.index = pd.DatetimeIndex(data1['Date'])

#once we set date as index, needs to del date column 
data1.drop(columns=['Date'], inplace=True)

data1.head()
#Resample() - resamples time series data based on specified frequency
data1['High'].resample('M').sum()

#current date frequncy is daily
data1['Close'].resample('W').mean()
#date_range() - creates array of datetime 

#creates 10 dates starting from 2020-01-05 with WEEK frequency
date1 = pd.date_range(start='2020-01-05', periods=10, freq='W')

#creates 10 dates ending date is 2020-03-10 with MONTH frequency
date2 = pd.date_range(end='2020-03-10', periods=10, freq='M')

#creates 10 dates ending date is 2020-03-10 with MONTH frequency
date3 = pd.date_range(start='2020-01-01', end='2020-06-01', freq='SM')

pd.DataFrame({"Date1": date1, "Date2": date2, "Date3": date3})
