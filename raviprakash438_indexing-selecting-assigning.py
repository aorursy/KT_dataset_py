import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from learntools.advanced_pandas.indexing_selecting_assigning import *
%matplotlib inline
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 10)
reviews.head()
check_q1(pd.DataFrame())
reviews.description
reviews.description[0]
reviews.iloc[0:1]
reviews.description[0:10]
reviews.iloc[[1,2,3,5,8],:]
reviews.iloc[[0,1,10,100],[0,5,6,7]]
reviews.iloc[0:100,[0,11]]
reviews[reviews.country=='Italy']
reviews[reviews.region_2.isna()==False]
reviews.points
wines=reviews[['points','winery']]
wines=wines.drop_duplicates('winery')
wines.head(1000)
wines.tail(1000)
ItalyWine=reviews[reviews.country=='Italy']
ItalyWine.points
reviews[((reviews.country=='Italy') |  (reviews.country=='France')) & (reviews.points>=90)]['country']
uswine=reviews[reviews.country=='US'].sort_values('price',ascending=False)
wine=uswine[['winery','price']]
wine=wine.drop_duplicates('winery')
wine=wine.head(10)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.barplot(wine.winery,wine.price,)
plt.show()