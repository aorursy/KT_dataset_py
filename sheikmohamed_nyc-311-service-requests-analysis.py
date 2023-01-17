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
df_NYC_dataset = pd.read_csv('../input/311_Service_Requests_for_2009.csv')
df_NYC_dataset.head()
#Shape before dropping nan values
df_NYC_dataset.shape
df_NYC_dataset.columns
#Complaint type Breakdown with bar plot to figure out majority of complaint types and top 10 complaints
df_NYC_dataset['Complaint Type'].value_counts().plot(kind='barh',alpha=0.6,figsize=(15,30))
plt.show()
#Have a look at the status of tickets
df_NYC_dataset['Status'].value_counts().plot(kind='bar',alpha=0.6,figsize=(7,7))
plt.show()
#Group dataset by complaint type to display plot against city
groupedby_complainttype = df_NYC_dataset.groupby('Complaint Type')
grp_data = groupedby_complainttype.get_group('HEATING')
grp_data.shape
#To get nan values in the entire dataset
df_NYC_dataset.isnull().sum()
#fix blank values in City column
df_NYC_dataset['City'].dropna(inplace=True)
#Shape after dropping nan values
df_NYC_dataset['City'].shape
#count of null values in grouped city column data
grp_data['City'].isnull().sum()
#fix those NAN with "unknown city" value instead
grp_data['City'].fillna('Unknown City', inplace =True)
#Scatter plot displaying all the cities that raised complaint of type 'HEATING'
plt.figure(figsize=(20, 15))
plt.scatter(grp_data['Complaint Type'],grp_data['City'])
plt.title('Plot showing list of cities that raised complaint of type HEATING')
plt.show()
#Find top 10 major complaint types and their counts
groupedby_complainttype['Complaint Type'].value_counts().nlargest(10)
#fix Location type those NAN with "unknown Location" value instead
df_NYC_dataset['Location Type'].fillna('Unknown Loc', inplace =True)
df_NYC_dataset['Location Type'].values
#count of null values in grouped location type column data
grp_data['Location Type'].isnull().sum()
#Plot Major complaint type Heating against location type to check for any pattern
plt.figure(figsize=(3,3))
plt.scatter(grp_data['Complaint Type'],grp_data['Location Type'])
plt.title='Plot complaint type Heating against location type'
plt.xlabel='Complaint Type'
plt.ylabel='Location Type'
plt.show()
#Plot below gives us a clear picture of the fact that all the complaints rasied of type "HEATING" in 2009 
#occured only in Residential Building! This shows that majority of complaints recorded was from Residential Building!