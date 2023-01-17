import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pyodbc # Used to link database to Python



# A simple MS Access 2010 database can be downloaded from: https://drive.google.com/drive/folders/0B2lDSJ7jLm_TNkpsVExONWVEWkU

db_file = r'.../UK Car Accidents 2005-2015.accdb' # database file location

conn = pyodbc.connect(r'Driver={Microsoft Access Driver (*.mdb, *.accdb)};DBQ=' + db_file) # Connect to the database

cursor = conn.cursor()



# Queries

q_from = '_Accidents'

q_select = '*'

q_where = 'Latitude<50 AND Longitude>-5.2'



# SQL command

sql = 'SELECT ' + q_select + ' FROM ' + q_from + ' WHERE ' + q_where + ';'



cursor.execute(sql)

data = cursor.fetchall()

data = [list(x) for x in data] # Convert to lists

df = pandas.DataFrame(data)

print(df)



cursor.close()

conn.close()



########################################################################



# Output should be as follows:

#              0       1      2         3          4   5   6   7   8   \

#0  200750A23O040  171700  15450 -5.186876  49.995330  50   2   1   1   

#1  200850AH2M003  171800  15370 -5.185435  49.994650  50   3   2   1   

#2  200950AH2M002  171630  15470 -5.187863  49.995483  50   3   2   2   

#3  201150AH2M001  171583  15483 -5.188525  49.995581  50   2   1   1   



#          9     ...      22 23  24 25  26  27  28  29  30         31  

#0 2007-11-19    ...       0  0   6  1   2   0   0   2   1  E01018881  

#1 2008-09-30    ...       0  0   1  2   2   0   0   2   1  E01018881  

#2 2009-04-21    ...       0  0   1 -1   1   0   0   2   1  E01018881  

#3 2011-07-06    ...       0  0   1  2   2   0   0   2   1  E01018881  



#[4 rows x 32 columns]
