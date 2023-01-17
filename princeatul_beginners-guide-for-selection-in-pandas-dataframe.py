# lets import pandas package
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
# reading the train file 
data = pd.read_csv('../input/train.csv')
# lets have a look at this dataset
data.head()
# select the name column
name = data['Name']
type(name)
name.head()
# select the name column but result should be a dataframe
name= data[['Name']]
type(name)
name.head()
# select name and sex of the passengers
name = data[['Name', 'Sex']]
name.head()
# select all the record of male passengers
male_record = data[(data['Sex']=='male')]
male_record.head()
# select all the rows where passengers is male and is 20 years or older
male_record = data[(data['Sex']=='male')&(data['Age']>= 20)]
male_record.head()
# select all the rows where age is null
age_null = data[(data['Age'].isnull())]
age_null.head()
# select all the rows where cabin is not null
cabin_value = data[(data['Cabin'].notnull())]
cabin_value.head()
data.head()
# Selecting any row using df.iloc
# select the first row 
data.iloc[0]
# first row can be accessed using 0 
# select the last row 
data.iloc[-1]
data.iloc[[0]]
# select top 10 rows
data.iloc[0:11]
# select first second, fourth and tenth rows
data.iloc[[0,1,3,9]]
# select first column

data.iloc[:,0]
# select starting three columns

data.iloc[:,0:3]
# select first, thrid and fifth columns
data.iloc[:,[0,2,4]]
# we can access any particluar cell using .iloc
data.iloc[0,0]
# Select 4th to 6th column in the first row
data.iloc[0,3:6]
# select 4th to 6th column for the first three row
data.iloc[0:3,3:6]
# select 2nd, 4th and 7th columns for the first three row
data.iloc[0:3,[1,3,6]]
## select 2nd, 4th and 7th columns for the first, three and fifth row
data.iloc[[0,2,4],[1,3,6]]
# df.iloc will return a pandas series if you are selecting only one row. It will return a pandas dataframe when multiple rows are selected.
type(data.iloc[0])
type(data.iloc[[0]])
type(data.iloc[:,0])
type(data.iloc[:,[0]])
data.head()
# we cannot use column name here 
# data.iloc[0,'Sex']
# above will throw an error
# select the rows where index = 0
data.loc[0]
# select the rows where index=0 and get the output in a dataframe
data.loc[[0]]
# select all the rows where index is less than or equal to 10 
data.loc[0:11]
# select all the rows where index is equal to 0, 5, 10, 20, 50
data.loc[[0,5,10,20,50]]
# select the cell where index = 0 and column = Name
data.loc[0,'Name']
# select the cell where index = 0 and columns are Name, Sex, Ticket
data.loc[0,['Name', 'Sex', 'Ticket']]
# select the cell where index in in the range from 0 to 11 and columns are Name, Sex, Ticket
data.loc[0:11,['Name', 'Sex', 'Ticket']]
# select all the data where index is equal to 0,5,10,20,50 and column names are Name, Sex, Ticket
data.loc[[0,5,10,20,50], ['Name', 'Sex', 'Ticket']]
# select all the data where index is in the range of 0 to 11 and all the columns from name to ticket
data.loc[0:11,'Name':'Ticket']