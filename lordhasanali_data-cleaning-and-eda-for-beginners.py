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
data = pd.read_csv('/kaggle/input/obesity-among-adults-by-country-19752016/data.csv', header=0, skiprows= range(1,3))
data.shape
data.head()
temp = data.T
new_header = temp.iloc[0] #grab the first row for the header
temp = temp[1:] #take the data less the header row
temp.columns = new_header #set the header row as the df header
temp.head(2)
##creating a new dataframe of COuntry and Obesity
new_data = pd.DataFrame(columns=['Country','Obesity%'])
for k,v in temp[temp.columns[1:]].items():
    new_data = new_data.append(pd.DataFrame({'Country':k ,'Obesity%': v.values}))
##Creating another Dataframe for year data
year_data = pd.DataFrame(columns=['Year','Sex'])
for year in range(2016,1974,-1):
    year_data = year_data.append(pd.DataFrame({'Year':year ,'Sex': ['Both sexes','Male','Female']}))
year_data.reset_index(drop=True, inplace=True)
###performing left join on the dataset
final_data = new_data.join(year_data, how='left')
###sorting the values on the basis of Country and Year
final_data.sort_values(by=['Country','Year'], inplace= True)
###cleaned dataset
final_data.head()