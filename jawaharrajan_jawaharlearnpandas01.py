import pandas as pd
pd.set_option('max_rows', 5)

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from creating_reading_writing import *
check_q1(pd.DataFrame())
import pandas as pd
df = pd.DataFrame([{'Apples': 30, 'Bananas':21}])
df
import pandas as pd
import numpy as np
data = np.array([['','Apples','Bananas'],
                ['2017 Sales',35,21],
                ['2018 Sales', 41, 34]])
df = pd.DataFrame(data=data[1:,1:],
                  index=data[1:,0],
                  columns=data[0,1:])
df
import pandas as pd
data = {'Flour': '4 cups','Milk': '1 cup', 'Eggs': '2 large', 'Spam': '1 can'}
pd.Series(data, name='Dinner')
import pandas as pd
df = pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv')
import pandas as pd
df = pd.ExcelFile('../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls')
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
import pandas as pd
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
q6_df.to_csv('cows_and_goats.csv')

import pandas as pd
import sqlite3
con = sqlite3.connect('../input/picthfork-data/database.sqlite')
df = pd.read_sql_query("SELECT * from reviews", con)
df.head()
con.close()