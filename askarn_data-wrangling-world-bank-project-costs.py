# Import required libraries
import numpy as np 
import pandas as pd 
import os
# Get current directory
os.getcwd()
# Load raw data, data is available: https://datacatalog.worldbank.org/dataset/world-bank-projects-operations 
data = pd.read_csv("../input/world-bank-projects-costs/World Bank Projects.csv",  sep='delimiter', header = None, 
                    skip_blank_lines=True,engine = 'python')
# Display full row
pd.options.display.max_colwidth = 1000
data.head()
# Get information about data
data.info()
# We need to split column of the data into multiple columns by comma
s = data[0].apply(lambda x: pd.Series(str(x).split(',')))
from IPython.display import display
pd.options.display.max_columns = None
s.head(2)


Project_Data = data[0].apply(lambda x: pd.Series(str(x).split(',')))
# Delete 0 row and columns beginning from 18
Project_Data = Project_Data.iloc[1:,:18]
Project_Data
Project_Data = Project_Data.drop(Project_Data.columns[[0,5,6,7,8,9,17]], axis = 1)
Project_Data
#  Define column names
Project_Data.columns = ["Region", "Country", "Product line", "Lending instrument", "Status", "Project name", "Approval date", "X1",
                "Closing date", "Sum of the project", "X2"]
# Remove unnecessary columns and unite "Sum of the project" and 'X2' columns
Project_Data.drop('X1', axis = 1, inplace = True)
Project_Data["Sum of the project"] = Project_Data["Sum of the project"] + Project_Data["X2"]
Project_Data.drop('X2', axis = 1, inplace = True)
# Remove duplicate country names in column 'Country'
a = Project_Data['Country'].apply(lambda x: pd.Series(str(x).split(';')))
Project_Data['Country'] = a[0] 
# Convert column 'Sum of the project' into numeric type
Project_Data['Sum of the project'] = Project_Data['Sum of the project'].str.replace('"', '')
Project_Data[['Sum of the project']] = (Project_Data[['Sum of the project']].apply(pd.to_numeric)*1000)
# extract only years from columns ''Approval date' and 'Closing date'
for i in ['Approval date', 'Closing date']:
    a = Project_Data[i].apply(lambda x: pd.Series(str(x).split('-')))
    Project_Data[i] = a[0] 
# Insert 'Undefined' where closing dates are not included
Project_Data[['Closing date']] = (Project_Data[['Closing date']].apply(pd.to_numeric))
Project_Data['Closing date'] = Project_Data['Closing date'].fillna('Undefined')
Project_Data
Project_Data.info()