import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
# Your code here
#ans=pd.DataFrame({'Apples':[30],'Bananas':[21]})
#check_q2(pd.DataFrame({'Apples':[30],'Bananas':[21]}))
check_q1(pd.DataFrame(
    {'Apples': [30], 'Bananas': [21]},
    index=[0]))
#print(answer_q1())
check_q2(pd.DataFrame(
    {'Apples': [35, 41], 'Bananas': [21, 34]},
    index=['2017 Sales', '2018 Sales']))
#print(answer_q2())
# Your code here
check_q3(pd.Series(['4 cups','1 cup','2 large','1 can'],
                  index=['Flour','Milk','Eggs','Spam'],
                  name='Dinner'))
                   #dtype='object'))
#print(answer_q3())
# Your code here
check_q4(pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv",index_col=0))
print(pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv"))
#print(answer_q4())
# Your code here 
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
check_q6(q6_df.to_csv("cows_and_goats.csv"))
# Your code here
import sqlite3
comm = sqlite3.connect("../input/pitchfork-data/database.sqlite")
check_q7(pd.read_sql_query("select *  from artists",comm))

# Your Code Here