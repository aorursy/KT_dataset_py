import pandas as pd
pd.set_option('max_rows', 5)

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from creating_reading_writing import *
check_q1(pd.DataFrame())
df=pd.DataFrame({"Apples":[30],"Bananas":[21]})
df

data={"Apples":[35,41],"Bananas":[21,34]}
pd.DataFrame(data,columns=["Apples","Bananas"])
pd.DataFrame(data, index=["2017 Sales","2018 Sales"])

sr=pd.Series(["4 cups","1 cup","2 large","1 can"], index=["Flour","Milk","Eggs","Spam"],name="Dinner")
sr
wi=pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv")
wi
wp=pd.read_excel("../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls","Pregnant Women Participating")
wp
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
# Your code here
# Your Code Here