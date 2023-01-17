import os          



os.getcwd()
os.chdir('/kaggle/')

        

os.getcwd()                     # Check the working directory again
os.listdir('/kaggle/input')
import pandas as pd



titanic_train = pd.read_csv('input/titanic/train.csv')    # Supply the file name (path)



titanic_train.head(6)                           # Check the first 6 rows
draft = pd.read_excel('input/draft2015/draft2015.xlsx', # Path to Excel file

                     sheet_name = 'draft2015')         # Name of sheet to read from



draft.head(6)                            # Check the first 6 rows
draft.to_csv("draft_saved.csv") 



os.listdir('/kaggle/')