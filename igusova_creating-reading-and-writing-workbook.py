import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
# Your code here
data = {'Apples':[30],
       'Bananas':[21]}
frame = pd.DataFrame(data)
check_q1(frame)
# Your code here
data1 = {'Apples':[35,41],
       'Bananas':[21,34]}
frame2 = pd.DataFrame(data1, index = ['2017 Sales', '2018 Sales'])
print(frame2)
check_q2(frame2)
# Your code here
data3 = ['4 cups','1 cup','2 large','1 can']
ser = pd.Series(data3, index = ['Flour','Milk','Eggs','Spam'])
ser.name = 'Dinner'
check_q3(ser)

# Your code here 
file = pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv', header = 0, index_col = 0)
print(file)
check_q4(file)
# Your code here
df = pd.read_excel('../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls', sheet_name='Pregnant Women Participating')
check_q5(df)
df
# Your code here
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
q6_df.to_csv('cows_and_goats.csv')
check_q6(q6_df.to_csv('cows_and_goats.csv'))
# Your Code Here