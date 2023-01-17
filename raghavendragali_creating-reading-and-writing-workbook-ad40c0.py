import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
# Your code here
pd.DataFrame({"Apples":[30],"Bananas":[21]})
# Your code here
pd.DataFrame({"Apples":[35,41],"Bananas":[21,34]},index=["2017 Sales","2018 Sales"])
# Your code here
Dinner=pd.Series(data=["4 cups","1 cup","2 large","1 can"],index=["Flour","Milk","Eggs","Spam"],name="Dinner")
Dinner
# Your code here 
filepath = "../input/wine-reviews/winemag-data_first150k.csv"
data=pd.read_csv(filepath)
data
# Your code here
data = pd.read_excel("../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls",sheet_name="Pregnant Women Participating")
data
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
# Your code here
q6_df.to_csv(path_or_buf="cows_and_goats.csv")
# Your Code Here