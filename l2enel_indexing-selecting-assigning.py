import pandas as pd
import seaborn as sns
from learntools.advanced_pandas.indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q4(q4)
reviews.head(1)
q1 = reviews.description
q2 = reviews.description[0]
q3 = reviews.iloc[0]
q4 = reviews.iloc[:10].description
q4
reviews.iloc[[1, 2, 3 , 5, 8]]
reviews.loc[[0, 1, 10, 100], ['country', 'province', 'region_1', 'region_2']]
reviews.loc[0:100, ['country', 'variety']]
reviews[reviews.country == 'Italy']
reviews[reviews.region_2.notnull()]
reviews.points
reviews.loc[0:1000, 'points']
x = reviews.loc[-1000:, 'points']
reviews[reviews.country == 'Italy'].points
france = reviews.query("country == 'France' & points >= 90")
italy = reviews.query("country == 'Italy' & points >= 90")

france.count()[0], italy.count()[0]