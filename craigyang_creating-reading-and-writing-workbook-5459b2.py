import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
a_1 = pd.DataFrame({
    'Apples':[30],
    'Bananas':[21]
})
check_q1(a_1)
a2 = pd.DataFrame({
    'Apples': [35,41],
    'Bananas':[21,34]
},index = ['2017 Sales','2018 Sales'])
check_q2(a2)
a3 = pd.Series ([
    '4 cups','1 cup','2 large','1 can'
],index = ['Flour','Milk','Eggs','Spam']
,name = 'Dinner')
check_q3(a3)
a4 = pd.read_csv
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
# Your code here
# Your Code Here