import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
d={
    'Apples': [30],
    'Bananas': [21]
}

testDFrame=pd.DataFrame(d)
check_q1(testDFrame)
d={
    'Apples': pd.Series([35, 41], index=['2017 Sales', '2018 Sales']),
    'Bananas': pd.Series([21, 34], index=['2017 Sales', '2018 Sales'])
}

testDFrame=pd.DataFrame(d)
check_q2(testDFrame)
testSeries=pd.Series({'Flour': '4 cups', 'Milk': '1 cup', 'Eggs': '2 large', 'Spam': '1 can'}, name='Dinner')
print(testSeries)
check_q3(testSeries)
testDFrame=pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv', index_col=0)
print(testDFrame)
check_q4(testDFrame)
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
q6_df.to_csv('cows_and_goats.csv')

check_q6(q6_df)
import sqlite3

connection=sqlite3.connect('../input/pitchfork-data/database.sqlite')
artist_details=connection.execute('select * from artists')
review_ids=[]
artists=[]
for artist_detail in artist_details:
    review_ids.append(artist_detail[0])
    artists.append(artist_detail[1])  
d={
    'reviewid': pd.Series(review_ids),
    'artist': pd.Series(artists)
}
testDFrame=pd.DataFrame(d)
check_q7(testDFrame)