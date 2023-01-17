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
# paste the copied filepath inside the function below to read in the csv file
covid_data = pd.read_csv("_____", 
                         index_col='SNo', parse_dates=True)
# We can change the index column to the first column by setting index_col = "first_column_name"
# parse_dates lets the notebook understand each row that contains a date

covid_data
covid_data.isnull().sum()
# Use the .head() function to retreive the first 5 rows of the datset
covid_data.head()
covid_data['Confirmed']
# Comment out the above and uncomment below and run the code to see that these acheive the same output
#covid_data.Confirmed
covid_data['Country/Region'].nunique()
covid_data['Country/Region'].unique()
covid_data['Deaths'].sum()
# sum the number of Confirmed cases below 
# sum the number of Recovered cases below 
# Using the sum() function, write a basic arithmetic function to calculate the number of active cases 

covid_data['Confirmed'].max()
# Using this method, you can retreive the entire row that corresponds to the value 
# that the Confirmed column takes on here
covid_data[covid_data['Confirmed'] == covid_data['Confirmed'].max()]
# Easy way to add a new column, displayed below
covid_data['Present_Confirmed'] = ____
covid_data
# Use code to answer the question: What is the biggest number of active cases right now?
_____
# covid_data['Present_Confirmed'].max()
# Use a method from above to retreive the row that corresponds to the biggest number of active cases
_____
# covid_data[covid_data['Present_Confirmed'] == covid_data['Present_Confirmed'].max()]
# Use similar method to retrieve all the data from Mainland China
_____

# china_data = covid_data[covid_data['Country/Region'] == 'Mainland China']