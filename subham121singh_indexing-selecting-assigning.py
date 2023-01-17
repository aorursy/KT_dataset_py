import pandas as pd
import seaborn as sns
from learntools.advanced_pandas.indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
# Your code here
r=reviews['description']
# Your code here
r[0]
# Your code here
reviews.iloc[0]
#check_q3(reviews.iloc[0])
# Your code here
import pandas as pd
pd.Series(reviews.description[:10]) 
# Your code here
reviews.iloc[[1,2,3,5,8]]
# Your code here
re=reviews[['country','province','region_1','region_2']]
dt=pd.DataFrame(re)
dt.iloc[[0,1,10,100]]
# Your code here
reviews[['country','variety']].iloc[0:101]
# Your code here
reviews[reviews['country']=='Italy']
re=reviews[reviews['region_2'] !='Nan']
reviews.query(re)['designation']
# Your code here
re=reviews['points']
# Your code here
re[:1000]
# Your code here
re[:1000:-1]
# Your code here
reviews[reviews['country']=='Italy'].points
reviews[reviews.country.isin(["Italy", "France"]) & (reviews.points >= 90)].country