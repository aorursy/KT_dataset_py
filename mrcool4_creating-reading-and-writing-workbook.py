import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
# Your code here
import pandas as pd
DF= pd.DataFrame({'Apples':[30],'Banana':[21]})
print(DF)
# Your code here
import pandas as pd
DF= pd.DataFrame({'Apples':[30,41],'Banana':[21,34]})
print(DF)
# Your code here
import pandas as pd
DF= pd.DataFrame({'Dinner':['Flour     4 cups','Milk       1 cup','Eggs     2 large','Spam       1 can']})
print(DF['Dinner'])
# Your code here 
import pandas as pd
WineReviews=pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv')
WineReviews
# Your code here
import pandas as pd
PublicAssistance=pd.read_excel('../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls')
PublicAssistance
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
# Your code here
import pandas as pd
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
q6_df.to_csv('cows_and_goats.csv')
# Your Code Here