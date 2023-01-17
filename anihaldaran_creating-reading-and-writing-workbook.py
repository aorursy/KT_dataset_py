import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
# Your code here 
import pandas as pd #for data frame
fruitData={'Apples':[30],
          'Bananas':[21]}
fd= pd.DataFrame(fruitData,
 columns=['Apples', 'Bananas'])
print(fd)
# Your code here
import pandas as pd #for data frame
fruitData2={'Apples':[35,41],
          'Bananas':[21,34],
           }
fd2= pd.DataFrame(fruitData2,
 columns=['Apples', 'Bananas'],
 index=['2017 Sales', '2018 Sales']) #index is for the name of each row
print(fd2)

# Your code here
recipeSeries = pd.Series(['4 cups', '1 cup', '2 large', '1 can'], index=['Flour', 'Milk', 'Eggs', 'Spam'], name='Dinner')
print(recipeSeries)
# Your code here 
countryFile=pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv") #reads the CSV file
countryFile.shape #shows how many rows and columns are present

print(countryFile.head())#prints the first 5 rows of the dataframe
# Your code here
Pregnancy=pd.read_excel("../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls")
Pregnancy.shape
print(Pregnancy.head())
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
# Your code here
q6_df.to_csv('cows_and_goats.csv')
# Your Code Here
#reads the SQL data
import sqlite3
conn = sqlite3.connect("../input/pitchfork-data/database.sqlite")
readSQL = pd.read_sql_query("SELECT * FROM artists", conn) #the name of the table is artists
readSQL.head()