import pandas as pd
import numpy as np
from learntools.advanced_pandas.renaming_and_combining import *

pd.set_option('max_rows', 5)
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
df = reviews.copy()
df.head()
# Your code here
check_q1(df.rename(columns={"region_1": "region", "region_2": "locale"}))

# Your code here

check_q2(df.rename_axis("wines", axis = "rows"))
#print(answer_q2())
gaming_products = pd.read_csv("../input/things-on-reddit/top-things/top-things/reddits/g/gaming.csv")
gaming_products['subreddit'] = "r/gaming"
movie_products = pd.read_csv("../input/things-on-reddit/top-things/top-things/reddits/m/movies.csv")
movie_products['subreddit'] = "r/movies"
# Your code here
#gaming_products.head()
#print(movie_products.columns)
check_q3(pd.concat([gaming_products, movie_products]))



powerlifting_meets = pd.read_csv("../input/powerlifting-database/meets.csv")
powerlifting_competitors = pd.read_csv("../input/powerlifting-database/openpowerlifting.csv")
# Your code here
powerlifting_meets.head()
powerlifting_competitors.head()
#check_q4(pd.merge(powerlifting_meets, powerlifting_competitors, how="left", left_on="MeetID", right_on="MeetID"))
#check_q4(pd.merge(powerlifting_meets, powerlifting_competitors, how="inner", left_on="MeetID", right_on="MeetID"))
powerlifting_meets.set_index("MeetID").join(powerlifting_competitors.set_index("MeetID"))
#print(answer_q4())
check_q4(pd.merge(powerlifting_meets, powerlifting_competitors, how="left", left_on="MeetID", right_on="MeetID").set_index("MeetID"))