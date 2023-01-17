import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
data = {'Apples':30,
       'Bananas': 21}
df = pd.DataFrame(data, columns = ['Apples','Bananas'])
check_q1(pd.DataFrame())
# Your code here
# Your code here
# Your code here 
# Your code here
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
# Your code here
# Your Code Here