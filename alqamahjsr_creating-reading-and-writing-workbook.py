import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
# Your code here
pd.DataFrame({'Apples':[30],'Bananas':[21]})

# Your code here
pd.DataFrame({'Apples':['35','41'],'Bananas':['21','34']},index=['2017 Sales','2018 Sales'])
# Your code here'
pd.Series(['4 Cups','1 Cup', '2 Large','1 Can'], index=['Flour','Milk','Eggs','Spam'],name='Dinner')
# Your code here 
wine_reviews=pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv")
wine_reviews.shape
wine_reviews.head()
wine_reviews=pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv",index_col=0)
wine_reviews.head()
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
print(q6_df)
# Your code here
wine_reviews.head().to_csv("Wine_Reviews.csv")
# Your Code Here
import sqlite3
conn=sqlite3.connect("../input/pitchfork-data/database.sqlite")
review_id=pd.read_sql_query("SELECT * from artists", conn)
review_id.head()
print(review_id)
conn=sqlite3.connect("review_id.sqlite")
review_id.head(10).to_sql("Review_Id",conn)