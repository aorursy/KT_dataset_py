# Code to import libraries 
import sys # import sys library to check the python version
print (sys.version_info)
import pandas as pd # import pandas library
import numpy as np # import numpy library
# Read excel file 
df = pd.read_excel('basic_indicators.xlsx')
df.head(20)
# Drop the unused columns that contains only 'NaN'
df.drop(df.columns.to_series()[17:24], axis=1, inplace=True)  # This column contains only 'NaN'
df.drop(df.columns[0], axis=1, inplace=True) # This column contains only 'NaN'
df.drop(df.columns[14], axis=1, inplace=True)  # This column contains 'NaN' and 'x', no actual meaning
df.head(20)
# Drop the first 3 rows 
df.drop(df.index[:4], inplace=True)
df.head(20)
# Drop the rows in the end of the file which is not needed in the output file
df.drop(df.index.to_series()[202:], inplace=True)   # From row 202 to the end of the file
# Rename the column names
for i in range(1, len(df.columns)):
    df.columns.values[i] = str(i-1)

df.columns.values[0] = 'Country Name'
df.head(20)
# Replace 'NaN' and '-' with a '-1', this helps to convert the strings to integer otherwise will triger errors
df = df.replace({np.nan: '-1', '–': '-1'}, regex=True)
df.head(20)
# Converting data type and round it to the nearest integer
data = df.copy()      # Make a copy of the dataframe before converting all the data
data.iloc[:,1:] = np.around(data.iloc[:,1:].astype(np.double)) # Rounding and converting, must use pn.around and np.double
data.head(20)
# Replace -1 with empty string 
data = data.replace(-1, '', regex=True)
# data = data.replace({-1: '', '-1':''}, regex=True)
data.head(20)
# Save dataframe to csv file
data.to_csv('basic_indicators.csv', sep='\t', encoding='utf-32', index=False)