import pandas as pd
pd.set_option('max_rows', 5)

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from creating_reading_writing import *
check_q1(pd.DataFrame())
import pandas as pd
pd.DataFrame({'Apples':[30],'Bananas':[21]})
import pandas as pd
pd.DataFrame({'Apples':[35,41],'Bananas':[21,34]},index=['2017 Sales','2018 Sales'])
import pandas as pd
pd.Series(['4 cops','1 cup','2 large','1 can'], index=['Flour','Milk','Eggs','Spam'])
import pandas as pd
df=pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv',index_col=0)
df.head()
import pandas as pd
df=pd.read_excel('../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls',sheet_name='Pregnant Women Participating')
df.head()
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
q6_df.head().to_csv("cows_and_goats.csv")
# Your Code Here