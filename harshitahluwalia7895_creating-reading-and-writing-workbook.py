import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
import pandas as pd
import numpy as np
df=pd.DataFrame({
    'Apple':[30],
    'Bananas':[21]
})
print(df)
data = {'Apples':[30], 'Bananas':[21]}
df=pd.DataFrame(data,index=['2017 Sales','2018 Sales'])
df
data={'Flour':'4 Cups','Milk':'1 Cup','Eggs':'2 Large','Spam':'1 can'}
Dinner=pd.Series(data)
print(Dinner)
import numpy as np
import pandas as pd
df=pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv')
import numpy as np
import pandas as pd
df=pd.read_excel('../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls')
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
q6_df.to_csv('cows_and_goats.csv')
df=pd.read_sql('../input/pitchfork-data/database.sqlite')