import pandas as pd
import seaborn as sns
from learntools.advanced_pandas.indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
# Your code here
desc = reviews['description']
check_q1(desc)
# Your code here
desc[0]
check_q2(desc[0])
# Your code here
first = reviews.iloc[0]
check_q3(first)
# Your code here
four = pd.Series(desc[:10])
check_q4(four)
# Your code here
five = reviews.iloc[[1,2,3,5,8]]
check_q5(five)
# Your code here
six = reviews.loc[[0,1,10,100], ['country', 'province', 'region_1', 'region_2']]
check_q6(six)
# Your code here
seven = reviews.loc[:99, ['country', 'variety']]
check_q7(seven)
# Your code here
eight = reviews.loc[reviews.country == 'Italy']
check_q8(eight)

# Your code here
nine = reviews.loc[reviews.region_2.isnull() == False]
check_q9(nine)
# Your code here
ten = reviews.loc[:,'points']
check_q10(ten)
# Your code here
eleven = reviews.loc[:999, 'points']
check_q11(eleven)
# Your code here
twelve = reviews.iloc[-1000:, 3]
check_q12(twelve)
# Your code here
thirteen = eight.loc[:, 'points']
check_q13(thirteen)
# Your code here
fourteen = reviews.loc[((reviews.country == 'France') | (reviews.country == 'Italy'))
                      & (reviews.points >= 90)]
fourteen = fourteen.country
check_q14(fourteen)