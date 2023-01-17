import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
q1_df = pd.DataFrame([[30,21]], columns=['Apples', 'Bananas'])
check_q1(q1_df)
q2_df = pd.DataFrame([[35,21], [41,34]],columns=['Apples','Bananas'],index=['2017 Sales', '2018 Sales'])
check_q2(q2_df)
q3_df = pd.Series({'Flour': '4 cups', 'Milk': '1 cup', 'Eggs': '2 large', 'Spam': '1 can'}, name='Dinner')
check_q3(q3_df)
q4_df = pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv', index_col=0)
check_q4(q4_df)
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
check_q6(q6_df)
q6_df.to_csv('cows_and_goats.csv')
from sqlalchemy import create_engine

con = create_engine('sqlite:///../input/pitchfork-data/database.sqlite')

q7_df = pd.read_sql_table('artists', con)
check_q7(q7_df)