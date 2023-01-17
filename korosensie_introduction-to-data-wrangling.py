#importing the pandas library

import pandas as pd



#loading the data as dataframe

dataframe=pd.read_csv("../input/titanic/test.csv")
#printing the first 5 rows of the data

dataframe.head()
#importing pandas library

import pandas as pd



#creating a dataframe

dataframe=pd.DataFrame()





#Adding columns

dataframe['Name']=['MS Dhoni','Percy Jackson']

dataframe['Occupation']=['Cricketer','Adventurer']

dataframe['Plays Football']=[True,False]
dataframe
#creating a series or a row

new_person=pd.Series(['Tom Cruise','Actor',False],index=['Name','Occupation','Plays Football'])
#appending the row to dataframe

dataframe.append(new_person,ignore_index=True)
#Proble,m---What are the characteristics of the DataFrame

#Solution--- We will use pandas library to describe the characteristics of DataFrame



#importing the pandas library

import pandas as pd



#loading the data as dataframe

dataframe=pd.read_csv("../input/titanic/test.csv")
#Displaying first 2 rows

dataframe.head(2)
#Displaying the last 2 rows of the data

dataframe.tail(2)
#To display the number of rows and columns of the dataframe

dataframe.shape
#To display the statistics of the dataframe

dataframe.describe()
#Problem---Slice the dataframe or select indiviual data

#Solution--- We will use loc and iloc for returning values



#importing pandas library

import pandas as pd



#Loading the titanic dataframe

dataframe=pd.read_csv("../input/titanic/test.csv")
#Selecting the first row

dataframe.iloc[0]
dataframe.iloc[1:3]
#setting index

dataframe=dataframe.set_index(dataframe['Name'])
#displaying dataframe

dataframe.head(2)
#Problem---Select some rows on some conditions

#Solution---We will use pandas library



#importing pandas library

import pandas as pd



#loading the titanic dataset

dataframe=pd.read_csv("../input/titanic/test.csv")
#selecting 2 rows where 'sex' is 'male'

dataframe[dataframe['Sex']=='male'].head(2)
#filtering further for age 

dataframe[(dataframe['Sex']=='male') & (dataframe['Age']>=50)].head(2)
#Problem---Replace the values in DataFrame

#Solution---We will use pandas replace



#importing pandas library

import pandas as pd



#loading the titanic dataset

dataframe=pd.read_csv("../input/titanic/test.csv")

#replacing values and displaying 2 rows

dataframe['Sex'].replace('male','man').head(2)

#Replacing multiple values at the same time

dataframe['Sex'].replace(['female','male'],['woman','man']).head()
#We can also rplace across the whole dataframe

dataframe.replace(3,'three').head(3)
#The replace also accepts regex(regular expressions)

dataframe.replace(r"Q","Queenstown",regex=True).head()
#Problem---rename the column in the dataframe

#Solution---We will use rename method



#importing pandas library

import pandas as pd



#loading the titanic dataset

dataframe=pd.read_csv("../input/titanic/test.csv")
#renaming the 2 column and showing 2 rows

dataframe.rename(columns={'PClass': 'Passenger Class', 'Sex': 'Gender'}).head(2)
#Problem---Calculate the statistic of the datframe

#Solution---We will use panda method



#importing pandas library

import pandas as pd



#loading the titanic dataset

dataframe=pd.read_csv("../input/titanic/test.csv")





# Calculating the statistics

print('Maximum:', dataframe['Age'].max())

print('Minimum:', dataframe['Age'].min())

print('Mean:', dataframe['Age'].mean())

print('Sum:', dataframe['Age'].sum())

print('Count:', dataframe['Age'].count())
#Problem---Select all the unique values in the column

#Solution---We will use pandas unique() method



#importing pandas library

import pandas as pd



#loading the titanic dataset

dataframe=pd.read_csv("../input/titanic/test.csv")
#Selecting all unique passenger classes

dataframe['Pclass'].unique()
#Alternate method is value_counts() it displays all unique values with their count

dataframe['Pclass'].value_counts()
#We can also count the number of unique values 

dataframe['Pclass'].nunique()
#Problem---Select all the missing values in the DataFrame

#Solution---We will use isnull and notnull method to return boolean values



#importing pandas library

import pandas as pd



#loading the titanic dataset

dataframe=pd.read_csv("../input/titanic/test.csv")



#Displaying the data of person whose age value is null

dataframe[dataframe['Age'].isnull()].head(3)
#Attempting to replace the NaN value

dataframe['Sex']=dataframe['Sex'].replace('male',NaN)



#we will get error because we need numpy library
#importing numpy library

import numpy as np



#Replacing male value as NaN

dataframe['Sex']=dataframe['Sex'].replace('male',np.nan)
dataframe.head(2)
#Problem---Delete a column in a dataframe

#solution---We will use pandas library drop



#importing pandas library

import pandas as pd



#loading the titanic dataset

dataframe=pd.read_csv("../input/titanic/test.csv")



#deleting the column age and displaying 2 rows

dataframe.drop('Age',axis=1).head(2)
#we can delete the column if it does not have name

dataframe.drop(dataframe.columns[1],axis=1).head(2)
#Problem---Delete one or more rows from the dataframe

#Solution---We will use the conditionals to delete the rows



#importing pandas library

import pandas as pd



#loading the titanic dataset

dataframe=pd.read_csv("../input/titanic/test.csv")
#Deleteing rows where sex is not male

dataframe[dataframe['Sex']!='male'].head(2)
#deleting specific row of a name

dataframe[dataframe['Name']!='Wilkes, Mrs. James (Ellen Needs)'].head(2)
#Deleting single row with given index

dataframe[dataframe.index!=0].head(2)
#Problem---Delete duplicate rows from the dataframe

#Solution---We will use the drop_duplicate to delete duplicate rows



#importing pandas library

import pandas as pd



#loading the titanic dataset

dataframe=pd.read_csv("../input/titanic/test.csv")
dataframe.drop_duplicates().head(2)
#checking how many rows deleted

print("Number Of Rows In The Original DataFrame:", len(dataframe))

print("Number Of Rows After Deduping:", len(dataframe.drop_duplicates()))
#deleting duplicates

dataframe.drop_duplicates(subset=['Sex'])
#We can keep the last row of duplicate data while deleting the default first row

dataframe.drop_duplicates(subset=['Sex'],keep='last')
#Problem---Group individual rows from the dataframe

#Solution---We will use the groupby to group rows



#importing pandas library

import pandas as pd



#loading the titanic dataset

dataframe=pd.read_csv("../input/titanic/test.csv")
#Grouping the rows by column Sex and calculating the mean()

dataframe.groupby('Sex').mean()
#grouping the rows and counting them

dataframe.groupby('Sex')['Name'].count()
#We can also do multiple grouping 

dataframe.groupby(['Sex','Pclass'])["Age"].mean()
#Problem---Group the rows by time

#solution---We will use resample to group the rows by time



#importing numpy and pandas library

import numpy as np

import pandas as pd



#creating a date range

time_index=pd.date_range('10/07/2020',periods=100000,freq='30S')



#creating dataframe

dataframe = pd.DataFrame(index=time_index)



#creating columns of random values

dataframe['Sale_amount']=np.random.randint(1,10,100000)

#Grouping the rows by week and calculating the total

dataframe.resample('W').sum()
dataframe.head()
#grouping the rows by 2 weeks and calculating the mean()

dataframe.resample('2W').mean()
#grouping the rows by Month and counting the rows

dataframe.resample('M').count()
#Problem---Apply some operation on every element of columns

#Solution---We can use pandas columns as any othe sequence in Python



#importing pandas library

import pandas as pd



#loading the titanic dataset

dataframe=pd.read_csv("../input/titanic/test.csv")
#printing the first 3 name in uppercase

for name in dataframe['Name'][0:3]:

    print(name.upper())
#We can also ue list comprehensions

[name.upper() for name in dataframe['Name'][0:3]]
#Problem---Apply any function to all the elements of the columns

#Solution---We will use apply() method to apply inbuilt or user-defined functions



#importing pandas library

import pandas as pd



#loading the titanic dataset

dataframe=pd.read_csv("../input/titanic/test.csv")

#creating a function to do uppercasing

def uppercase(x):

    return x.upper()



#Applying funtion to the columns

dataframe['Name'].apply(uppercase)[:3]
#Problem---Apply functions to each group

#solution---We will use groupby and apply



#importing pandas library

import pandas as pd



#loading the titanic dataset

dataframe=pd.read_csv("../input/titanic/test.csv")
#applying count function to group

dataframe.groupby('Sex').apply(lambda x:x.count())
#Problem---Concatenate two dataframes

#Solution---We will use concat to concatenate to dataframes



#importing pandas library

import pandas as pd





#Creating dataframe

data_a={'id': ['1', '2', '3'],

'first': ['Levi', 'Eren', 'Eren'],

'last': ['Ackerman', 'Yeager', 'Kruger']}



dataframe_a=pd.DataFrame(data_a,columns=['id', 'first', 'last'])



#creating second dataframe

data_b = {'id': ['4', '5', '6'],

'first': ['Armin', 'Hange', 'Erwin'],

'last': ['Arlet', 'Zoe', 'Smith']}

dataframe_b = pd.DataFrame(data_b, columns = ['id', 'first', 'last'])
#concatenating two dataframes by rows

pd.concat([dataframe_a,dataframe_b],axis=0)
#concatenating two dataframes by columns

pd.concat([dataframe_a, dataframe_b], axis=1)
#we can use append() method to concatenate series to the dataframe

new_person =pd.Series(['7','Historia','Reiss'],index=['id','first','last'])



#appending the new_person

dataframe_a.append(new_person,ignore_index=True)
#Problem---Merge two dataframes

#Solution---We will use merge to merge dataframes





# Importing pandas library

import pandas as pd



# Creating DataFrame

employee_data = {'employee_id': ['1', '2', '3', '4'],

'name': ['Amy Jones', 'Allen Keys', 'Alice Bees',

'Tim Horton']}

dataframe_employees = pd.DataFrame(employee_data, columns = ['employee_id',

'name'])



# Creating DataFrame

sales_data = {'employee_id': ['3', '4', '5', '6'],

'total_sales': [23456, 2512, 2345, 1455]}

dataframe_sales = pd.DataFrame(sales_data, columns = ['employee_id',

'total_sales'])
#Inner merging dataframe 

pd.merge(dataframe_employees,dataframe_sales,on='employee_id')
#Outer Merging dataframes

pd.merge(dataframe_employees,dataframe_sales,on='employee_id',how='outer')
#Left Merging dataframes

pd.merge(dataframe_employees,dataframe_sales,on='employee_id',how='left')
#Right Merging dataframes

pd.merge(dataframe_employees,dataframe_sales,on='employee_id',how='right')
#we can specify thr column's name of dataframe to merge on

pd.merge(dataframe_employees,

        dataframe_sales,

        left_on='employee_id',

        right_on='employee_id')