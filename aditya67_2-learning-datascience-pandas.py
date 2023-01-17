import numpy as np
import pandas as pd
#Creating Series (Columns)
series1 = pd.Series(data=[1,2,3,4,5], index=list('abcde'))
series1
series2 = pd.Series(data=[10,20,30,40,50], index=list('abcde'))
series2
#Creating a DataFrame from the Series
df = pd.DataFrame({'A':series1,'B':series2})
df
#Random DataFrame 
pd.DataFrame(data = np.random.randint(1,10,size=(10,10)), index=list('ABCDEFGHIJ'), columns=list('abcdefghij'))
#Read CSV file
data_hr = pd.read_csv('../input/HR_comma_sep.csv.txt')

#Information about the dataset
data_hr.info()
#Read a JSON file
pd.read_json('../input/movie.json.txt')
#Connecting to Database and fetching Data
#Commented out
'''import MySQLdb
mysql_cn= MySQLdb.connect(host='myhost',
          port=3306,user='myusername', passwd='mypassword', 
          db='information_schema')
df_mysql = pd.read_sql('select * from VIEWS;', con=mysql_cn)'''
#Viewing Data
data_hr.head() #First 5 rows
data_hr.tail() #Last 5 rows
data_hr.columns
data_hr.describe()
data_hr.salary.value_counts()
#Categorise the object datatypes into a subset of main df.
cat_col_data = data_hr.select_dtypes('object')
cat_col_data.head()
#Rename a column
data_hr.rename(columns = {'sales' : 'department'},inplace=True)
data_hr.head()
#Check values of custom columns.
data_hr[['satisfaction_level','last_evaluation','number_project']].head()
#Particular Column
data_hr.satisfaction_level[:5]
data_hr['satisfaction_level'][:5]
movie_data = pd.read_json('../input/movie.json.txt')
movie_data
#Access using indexes using loc. (Not integers)
movie_data.loc['Raging Bull']
movie_data.loc['Goodfellas' : 'Scarface']
#For using numerical Indexes. We use iloc in place of loc (zero indexing)
movie_data.iloc[3]
movie_data.iloc[2:5]
#Using Conditions
movie_data['Bill Duffy']
movie_data[movie_data['Bill Duffy']>3]
#Our Data has many null values whoch migh not be useful.
movie_data
movie_data['Bill Duffy'].notnull() #Returns boolean for notnull()
#All rows for which Bill Duffy Column is not null.
movie_data[movie_data['Bill Duffy'].notnull()]
#All rows for which Bill Duffy is null.
movie_data[movie_data['Bill Duffy'].isnull()]
titanic_data = pd.read_csv('../input/titanic-train.csv.txt')
titanic_data.info()
#We can drop Cabin Column as it has 204 non-null values out of 891
titanic_data.drop(['Cabin'],inplace=True,axis=1)
titanic_data.info()
#Dropping all rows with missing or null values
#This will not cahnge the dataframe as we do not have inplace=True
titanic_data.dropna().info()
titanic_data.info()
titanic_data.dropna(subset=['Embarked','Age']).info()
titanic_data.info()
#We are filling Age with 0 and Embarked with Unknown for all NA values in these columns.
titanic_data.fillna({'Age':0,'Embarked':'Unknown'}).info()
df = pd.DataFrame({'A':[1,1,3,4,5,1], 'B':[1,1,3,7,8,1], 'C':[3,1,1,6,7,1]})
df
#We have 1st Row and 5th Row as Duplicates.
df.duplicated() #Returns Boolean True if duplicated
df[df.duplicated()]
#Creating a subset with required columns as labels
df[df.duplicated(subset = ['A','B'])]
titanic_data_age = titanic_data[titanic_data.Age.notnull()]
#Created a new column Age_category and mapped it using lambda.
titanic_data['Age_category'] = titanic_data.Age.map(lambda x : 'Kid' if x < 18 else 'Adult')
titanic_data.head(10)
#Calling a function using apply
def fare(e):
    if e.Sex == 'male':
        return e.Fare * 2
    else:
        return e.Fare
titanic_data[['Fare','Sex']].head()
#After we apply the function 'fare', we can see fare is doubled for male.
titanic_data.apply(fare,axis=1)[:5]
#GroupBy
#We group the df with 'Sex' and calculate the mean of Age.
titanic_data.groupby(['Sex']).Age.mean()
titanic_data.groupby(['Sex']).Age.agg(['mean','min','max'])
#Contains a specific substring in any value of column
titanic_data[titanic_data.Name.str.contains('Mrs')]
titanic_data[titanic_data.Name.str.contains('Mrs')].shape
