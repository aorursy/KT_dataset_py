import pandas as pd
from learntools.advanced_pandas.grouping_and_sorting import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
check_q1(pd.DataFrame())
reviews.head()
# Your code here

common_wine_reviewers2 = reviews.groupby('taster_twitter_handle').taster_twitter_handle.count()
common_wine_reviewers2

check_q1(common_wine_reviewers2)
# Your code here
two = reviews.groupby('price').points.max()
two

check_q2(two)
# Your code here
three = reviews.groupby('variety').price.agg([min, max])
three

check_q3(three)
# Your code here
four = reviews.groupby('taster_name').points.mean()
four
check_q4(four)
# Your code here
five = reviews.groupby('variety').price.agg([min, max])
#five
five = five.sort_values(by=['min', 'max'], ascending=False)
#five

check_q5(five)
# Your code here
six = reviews.groupby(['country', 'variety']).variety.count()

six = six.sort_values(ascending=False)
six

check_q6(six)