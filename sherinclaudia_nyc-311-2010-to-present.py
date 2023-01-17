# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
%matplotlib inline
#Import a 311 NYC service request
df = pd.read_csv('../input/311_Service_Requests_from_2010_to_Present.csv')
#Shape dataset before dropping nan values
df.shape
#Display column names
df.columns
#Status of tickets
df['Status'].value_counts().plot(kind='bar',alpha=0.6,figsize=(5,5))
plt.show()
#To get nan values in the entire dataset
df.isnull().sum()
#fix blank values in City column
df['City'].dropna(inplace=True)
#fix blank values in Complaint Type column
df['Complaint Type'].dropna(inplace=True)
#To get nan values in the entire dataset
df.isnull().sum()
#Group dataset by complaint type to display
groupedby_complainttype = df.groupby('Complaint Type')
#Find the top complaint type 
groupedby_complainttype['Complaint Type'].value_counts().nlargest(1)
#Group dataset by major complaint type to display
grp_data = groupedby_complainttype.get_group('Blocked Driveway')
grp_data.shape
#count of null values in grouped city column data
grp_data['City'].isnull().sum()
#count of null values in grouped Location Type column data
grp_data['Location Type'].isnull().sum()
#count of null values in grouped Complaint Type column data
grp_data['Complaint Type'].isnull().sum()
#fix those NAN with "unknown city" value instead
grp_data['City'].fillna('Unknown City', inplace =True)
#fix those NAN with "Location Type" value instead
grp_data['Location Type'].fillna('Unknown Location', inplace =True)
#Scatter plot displaying all the cities that raised complaint of type 'Blocked Driveway'
plt.figure(figsize=(20, 15))
plt.scatter(grp_data['Complaint Type'],grp_data['City'])
plt.title('Plot showing list of cities that raised complaint of type Blocked Driveway')
plt.show()
#Display array of Location Types
df['Location Type'].values
#Display complaint type and city together
df[['Complaint Type','City']]
#Find the top 10 complaint types 
groupedby_complainttype['Complaint Type'].value_counts().nlargest(10)
#Plot a bar graph of count vs. complaint types
df['Complaint Type'].value_counts().plot(kind='barh',alpha=0.6,figsize=(5,5))
plt.show()
#Display the major complaint types and their count
groupedby_complainttype['Complaint Type'].value_counts().nlargest(50)

