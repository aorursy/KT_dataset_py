# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
bengaluru=pd.read_csv('/kaggle/input/bengaluru-house-price-data/Bengaluru_House_Data.csv')
bengaluru.sample(10)
#Data Preprocessing

# 1.Gathering Data [Done]

# 2.Accessing Data

#       a.Incorrect data types[area_type,availability,size,total_sqft,bath,balcony]

#       b.Missing values[location,size,society,bath,balcony]

#       c.Outliers present[size,bath]
#Shape of the data 

bengaluru.shape
#Data type of columns

bengaluru.info()
#To find the total number of null elements in each column

bengaluru.isnull().sum()
#Mathematical columns

bengaluru.describe().T
# 1. Location
bengaluru['location'].value_counts()
#Filling the NaN element with the majority ones

bengaluru['location']=bengaluru['location'].fillna('Whitefield')
bengaluru.isnull().sum()
#The 'locations' which are in the data less than 10 times, just referring them as 'Others'
bengaluru.location=bengaluru.location.apply(lambda x: x.strip())

location = bengaluru['location'].value_counts()

location
location.values.sum()
len(location[location<=10])
location_less_than_10=location[location<=10]

location_less_than_10
bengaluru.location=bengaluru.location.apply(lambda x: 'Others' if x in location_less_than_10 else x)

bengaluru.location
bengaluru['location'].value_counts()
# 2. Area Type



#Converting the data type into 'category' because only four types are present['Super built-up Area','Built-up Area','Plot Area','Carpet Area']
bengaluru['area_type']=bengaluru['area_type'].astype('category')
bengaluru.info()
bengaluru.isnull().sum()
# 3. Size
bengaluru['size']
#Removing BHK,bedroom and RK from the 'size' column

bengaluru['size']=bengaluru['size'].str.split().str[0]

bengaluru['size']
bengaluru['size'].value_counts()
#Dropping the NaN elements

bengaluru.dropna(subset=['size'],inplace=True)
#Converting the data type of 'size' column to int32 to remove the decimal point

bengaluru['size']=bengaluru['size'].astype('int32')
import seaborn as sns
import matplotlib.pyplot as plt
sns.boxplot(bengaluru['size'])
#Removing the outliers which are more than 10 beacuse they are insane

bengaluru=bengaluru.drop(bengaluru[bengaluru['size']>10].index)
#Converting the data type to category because there are only 10 categories present now

bengaluru['size']=bengaluru['size'].astype('category')
bengaluru.info()
# 4. Society
bengaluru['society'].value_counts()
#Filled the NaN values with 'Information not Available'

bengaluru['society']=bengaluru['society'].fillna('Information Not Available')

bengaluru['society'].value_counts()
#The 'society' which are in the data less than 10 times, just referring them as 'Others'
bengaluru.society.apply(lambda x: x.strip())

society = bengaluru['society'].value_counts()

society
society.values.sum()
len(society[society<=10])
society_less_than_10=society[society<=10]

society_less_than_10
bengaluru.society=bengaluru.society.apply(lambda x: 'Others' if x in society_less_than_10 else x)

bengaluru.society
bengaluru['society'].value_counts()
bengaluru.info()
# 5.Availabilty
bengaluru['availability'].value_counts()
#Referring the dates as 'Available Soon' to convert the column into consistent data type 
mask1=[i for i in bengaluru['availability'] if i not in ['Ready To Move','Immediate Possession']]

bengaluru['availability']=bengaluru['availability'].replace(mask1,'Available Soon')

bengaluru['availability']
#Converting the data type of 'availabilty' column to category because there are 3 categories['Available Soon','Immediate Possession','Ready To Move']

bengaluru['availability']=bengaluru['availability'].astype('category')

bengaluru['availability']
bengaluru.info()
# 6.Total Square feet
#Doing Average of the square feets which are given in the range (for example 2400-2600)

def sqft(x):

    t=x.split('-')

    if len(t)==2:

        return (float(t[0])+float(t[1]))/2

    try:

        return x

    except:

        return None
bengaluru.total_sqft=bengaluru.total_sqft.apply(sqft)
bengaluru.total_sqft
#Converting different units to square feet 

def change_to_sqft(x):

    if("Sq. Meter" in str(x)):

        y=x.split("S")

        z=float(y[0])*10.76

        return z

    

    elif("Sq. Yards" in str(x)):

        y=x.split("S")

        z=float(y[0])*9

        return z

    

    elif("Guntha" in str(x)):

        y=x.split("G")

        z=float(y[0])*1088.98

        return z

    

    elif("Acres" in str(x)):

        y=x.split("A")

        z=float(y[0])*43560

        return z

    

    elif("Perch" in str(x)):

        y=x.split("P")

        z=float(y[0])*272.25

        return z

    

    elif("Cents" in str(x)):

        y=x.split("C")

        z=float(y[0])*435.6

        return z

    

    elif("Grounds" in str(x)):

        y=x.split("G")

        z=float(y[0])*2400

        return z

    

    else:

        return x
bengaluru['total_sqft']=bengaluru['total_sqft'].apply(change_to_sqft)
#Converting the data type of 'total_sqft' in to float64

bengaluru['total_sqft']=bengaluru['total_sqft'].astype('float64')
bengaluru['total_sqft']
bengaluru.info()
bengaluru.sample(10)
# 7.Balcony
bengaluru['balcony'].value_counts()
#Filling the NaN values to the mode of the column which is 2.0

bengaluru["balcony"]=bengaluru["balcony"].fillna(bengaluru["balcony"].mode()[0])
bengaluru["balcony"]=bengaluru["balcony"].astype('int32')
#Converting the data type to 'category' because there are 3 categories present now

bengaluru["balcony"]=bengaluru["balcony"].astype('category')
bengaluru.info()
bengaluru.sample(5)
# 8.Bath
bengaluru['bath'].value_counts()
#Filling the NaN values to the mode of the column which is 2.0

bengaluru["bath"]=bengaluru["bath"].fillna(bengaluru["bath"].mode()[0])
#Converting the data type to 'int32' 

bengaluru["bath"]=bengaluru["bath"].astype('int32')
bengaluru.info()
sns.boxplot(bengaluru['bath'])
#Removing the outliers which are insane

bengaluru=bengaluru.drop(bengaluru[bengaluru['bath']>5].index)
bengaluru["bath"]=bengaluru["bath"].astype('int32')
bengaluru['bath']
#Converting the data type to 'category' because there are 5 categories present now

bengaluru["bath"]=bengaluru["bath"].astype('category')
bengaluru.info()
bengaluru.sample(5)
#Conclusion



#The dataframe is cleaned and can be used for further analysis(EDA).