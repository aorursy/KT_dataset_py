import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
import pandas as pd
data = {'Apples': [30], 'Bananas': [21]}
df = pd.DataFrame(data=data)
print(df)
check_q1(df)
import pandas as pd
data = {'Apples':[35,41], 'Bananas': [21,34]}
df = pd.DataFrame(data=data, index=['2017 Sales', '2018 Sales'])
print(df)
check_q2(df)
import pandas as pd
data = ['4 cups', '1 cup','2 large', '1 can']
s = pd.Series(data=data, index=['Flour', 'Milk','Eggs', 'Spam'], name='Dinner')
print(s)
check_q3(df)
import pandas as pd
df = pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv', index_col=0)
df.head()
check_q4(df)
import pandas as pd
df= pd.read_excel('../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls', 'Pregnant Women Participating')
df.head()
check_q5(df)
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])

q6_df.to_csv('cows_and_goats.csv')
check_q6(q6_df.to_csv('cows_and_goats.csv'))
import sqlite3
import pandas as pd

dat = sqlite3.connect('database.sqlite')
query = dat.execute("SELECT * From artists")
cols = [column[0] for column in query.description]
results= pd.DataFrame.from_records(data = query.fetchall(), columns = cols)
