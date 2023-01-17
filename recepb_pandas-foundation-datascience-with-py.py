import numpy as np

#NumPy is a python library used for working with arrays.

#It also has functions for working in domain of linear algebra, fourier transform, and matrices.

#We have lists that serve the purpose of arrays, but they are slow.NumPy aims to provide an array object that is up to 50x faster that traditional Python lists.



import pandas as pd 

#Why pandas: you want to explore a dataset stored in a CSV on your computer. Pandas will extract the data from that CSV into a DataFrame — 

#a table, basically — then let you do things like:

#Calculate statistics and answer questions about the data, like: What's the average, median, max, or min of each column?

#Does column A correlate with column B?

#What does the distribution of data in column C look like?

#Clean the data by doing things like removing missing values and filtering rows or columns by some criteria

#Visualize the data with help from Matplotlib. Plot bars, lines, histograms, bubbles, and more.

#Store the cleaned, transformed data back into a CSV, other file or database



import os

#The OS module in python provides functions for interacting with the operating system.

#This module provides a portable way of using operating system dependent functionality.

#The *os* and *os.path* modules include many functions to interact with the file system.



import matplotlib.pyplot as plt

#Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python.



import seaborn as sns



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

#UTF-8 is a variable-width character encoding standard 

#that uses between one and four eight-bit bytes to represent all valid Unicode code points.



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.       
# Import pandas library 

import pandas as pd 

# initialize list of lists 

data = [['tom', 10], ['nick', 15], ['juli', 14]] 

# Create the pandas DataFrame 

df = pd.DataFrame(data, columns = ['Name', 'Age']) 

# print dataframe. 

df 

# DataFrame from dict narray / lists  

# By default addresses. 

import pandas as pd 

# intialise data of lists. 

data = {'Name':['Xavi', 'Jean', 'Muller', 'Jarkovski'], 'Age':[20, 21, 19, 18]} 

# Create DataFrame 

df = pd.DataFrame(data) 

# Add new columns

df["City"] = ["Barcelona","Paris","Berlin","Warsaw"]

df
# Python code demonstrate how to create Pandas DataFrame by lists of dicts. 

import pandas as pd 

# Initialise data to lists. 

data = [{'a': 1, 'b': 2, 'c':3}, {'a':10, 'b': 20, 'c': 30}] 

# Creates DataFrame. 

df = pd.DataFrame(data) 

# Print the data 

df 
data=pd.read_csv("/kaggle/input/craigslist-carstrucks-data/vehicles.csv")

data.head()
data = data.drop(labels= ['id', 'url', 'region_url', 'size','image_url','state','title_status','lat','vin','region'], axis=1)

data.head()
# Python program to demonstrate creating pandas Datadaframe from lists using zip. 



Name = ['Jane', 'Carl', 'Mark', 'Alex'] 

Age = [25, 30, 26, 22] 



# get the list of tuples from two lists. and merge them by using zip(). 

list_of_tuples = list(zip(Name, Age)) 



# Converting lists of tuples into pandas Dataframe. 

df = pd.DataFrame(list_of_tuples, columns = ['Name', 'Age']) 



# Print data. 

df 

data1 = data.loc[:,["long","year","price"]]

data1.plot(subplots = True)

plt.show()
data1.plot(kind = "scatter",x="year",y = "price")

plt.show()
time_list = ["1992-03-08","1992-04-12"]

print(type(time_list[1])) # As you can see date is string however we want it to be datetime object

datetime_object = pd.to_datetime(time_list)

print(type(datetime_object))

#parse_dates(boolean): Transform date to ISO 8601 (yyyy-mm-dd hh:mm:ss ) format
data2 = data.head()

date_list = ["10.07.2020","20.08.2020","30.09.2020","10.10.2020","8.11.2020"]

datetime_object = pd.to_datetime(date_list)

data2["DateTime"] = datetime_object

# lets make date as index

data2= data2.set_index("DateTime")

data2 
print(data2.loc["30.09.2020"])
data2.resample("A").mean()
data2.resample("M").mean()
# Storing values such as Date, Time Worked, and Money Earned in a DataFrame.

#All these are just raw data, which is later stored in pandas DataFrame and allocated to a variable df.To do this just use 

#“pd.DataFrame” and pass in all the data, by doing this the pandas will automatically convert the raw data into a DataFrame.



df = pd.DataFrame({'Date':['11/05/19', '12/05/19', '19/05/19', '25/05/19', '26/05/19', '1/06/19'],'Time Worked': [3, 3, 4, 3, 3, 4],'Money Earned': [33.94, 33.94, 46, 33.94, 33.94, 46]})



# Head displays only the top 5 rows from the data frame.

df.head()
# Adding more rows to the existing DataFrame (updating the rows of the DataFrame)

# inside the append function we have some other parameters as ignore_index = True, this prevents the data frame from appending

# new index, so in this example all the index are in a continuous fashion (incrementing), and the next parameter 

# is a sort = False this is because we don’t want to sort the data according to the index, otherwise our data would be completely a mixture.



# Adding more rows

df2 = pd.DataFrame({'Date': ['10/06/19', '12/06/19', '14/06/19'],

                    'Time Worked': [3, 4, 3],

                    'Money Earned': [33.94, 46, 33.94]})



df = df.append(df2, ignore_index=True, sort = False)

df
#Calculating the sum of Money earned and the total duration worked



Total_earnings = df['Money Earned'].sum()

Total_time = df['Time Worked'].sum()

print('You have earned total of ====>' ,round(Total_earnings),'Euro')

print(' — — — — — — — — — — — — — — — — — — — — — — — — — — — ')

print('You have worked for a total of ====>', Total_time, 'hours')
#Plotting the bar graph with Total duration vs Money Earned

#pandas have a plot () which will help you plot a graph up to a certain extent.



# Plotting a bar graph using pandas library.

df.plot(x ='Date', y='Money Earned', kind = 'bar')

plt.show()
#Including a search option in order to search for the respective dates worked.



#In real time projects, this is a handy feature where you often have to search for the data, there you cannot search it manually, 

#so with the help of this below code snippet you just get the job done. 

#str.contains() searches for the entered date and them displays the date and the corresponding values from the data frame.



# Including a search option.

date = input('Enter the date you want to search ===> ')

df[df['Date'].str.contains(date)]
# Adding the payroll option



# Function payroll contains calculation of total money.

def payroll():

 name = input('Enter the name of the employee: ==> ')

 hours = int(input('Enter the hours worked by the employee ==>'))

 rate = float(input('Enter the pay rate for one hour ==> '))

 total_money = hours * rate 

 print('The total money earned by ', name, 'for working ', hours,  'hours', 'is ===> ', round(total_money), 'Euro')

payroll()