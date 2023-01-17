import pandas as pd
from learntools.advanced_pandas.grouping_and_sorting import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
#reviews.head()
#reviews.columns
#Assuming we like to get number of wiineries by country
reviews.groupby('country').winery.value_counts().sort_values(ascending=False)
# Your code here
#print(answer_q1())
reviews.groupby('taster_twitter_handle').taster_twitter_handle.count()
# Your code here
reviews.groupby('price').points.max().sort_index()

# Your code here
reviews.groupby('variety').price.agg([min,max])

# Your code here
reviews.groupby('taster_name').points.mean()
# Your code here
minMax = reviews.groupby('variety').price.agg([min,max]).sort_values(by=['min','max'],ascending=False)
minMax

# Your code here

reviews['n'] = 0
reviews.groupby(['country','variety']).n.count().sort_values(ascending=False)
