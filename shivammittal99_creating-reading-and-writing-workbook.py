import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
# Your code here
pd.DataFrame({'Apples': [30], 'Bananas': [21]})
# Your code here
pd.DataFrame({'Apples': [35, 41], 'Bananas': [21, 34]}, index = ['2017 Sales', '2018 Sales'])
# Your code here
pd.Series({'Flour': '4 cups', 'Milk': '1 cup', 'Eggs': '2 large', 'Spam': '1 can'}, name = 'Dinner')
# Your code here 
pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv')
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
# Your code here
q6_df.to_csv('cows_and_goats.csv')
# Your Code Here
print(answer_q7())