import pandas as pd
pd.set_option('max_rows', 5)

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from creating_reading_writing import *
check_q1(pd.DataFrame())
# my code here
pd.DataFrame({'Apples':[30], 'Bananas':[21]}, index = [0])
# my code here
pd.DataFrame({'Apples':[35,41], 'Bananas':[21,34]}, index= ["2017 Sales", '2018 Sales'])
# my code here
pd.Series(['4 cups', '1 cup', '2 large', '1 can',], index = ['Flour', 'Milk', 'Eggs', 'Spam'], name= 'Dinner')
# my code 
wine_review=pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv",index_col=0)
# my code here
preg=pd.read_excel("../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls",sheet_name="Pregnant Women Participating")
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
# Your code here
# Your Code Here