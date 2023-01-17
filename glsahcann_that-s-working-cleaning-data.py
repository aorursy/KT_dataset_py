# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/athlete_events.csv')  #we got the data we wanted to work
data.shape  #How many columns and lines does our dataframe consist of?
data.info()  #gives columns specifications
data.columns #gives data columns name(features)
data.head()  #show the top five entries in the database
#data.head(10)  -> if you want top ten entries in the database , enough to write in brackets
data.tail()  #gives the last five entries in the database
#data.tail(20)  ->if you want last twenty entries in the database , enough to write in brackets
data.describe()  #gives information about numeric values
data.boxplot(column='Height', by ='Sex')
data_1 = data.head()  #create new data consisting of the top five entries in the database
data_1
melted = pd.melt(frame=data_1, id_vars = 'Name', value_vars = ['Height','Weight'])  #create list of values of height and weight of each name column
melted
melted.pivot(index = 'Name', columns = 'variable',values='value')
#Vertical  
data1 = data.head()
data2 = data.tail()
concat_data = pd.concat([data1,data2],axis =0, ignore_index =True)  #adds row in data frame
concat_data
#Horizontal
data1 = data['Sport'].head()
data2= data['Event'].head()
concat_data = pd.concat([data1,data2],axis =1)  #combine the two selected columns
concat_data
data.dtypes  #gives feature type
data['Year'] = data['Year'].astype(object)  #change of feature type
data.dtypes
#data['Age'] = data['Age'].astype(int)   -> gives an error "Cannot convert non-finite values (NA or inf) to integer" 
data.info()   # As you can see there are 271116 entries and age column has 261642 value non-null
data['Age'].value_counts(dropna =False)  # As you can see age column has 9474 null value.
data1 = data.copy()  #create new data with same data in memory
data1['Age'].dropna(inplace = True)   #drop the null value in the data1
#Lets check
data1['Age'].value_counts(dropna = False)  #see no null value
#You can testing with Assert Statement
assert 1==1  # return nothing because it is true
#assert 1==2   -> gives an error because it is false
assert  data1['Age'].notnull().all()   #returns nothing because we drop nan values
data1['Medal'].fillna('NOT WİN',inplace = True)
assert  data1['Medal'].notnull().all()   #returns nothing because we fill nan values
#Lets check
data1['Medal'].value_counts(dropna = False)  #see no null value