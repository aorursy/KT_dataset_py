import pandas as pd
import seaborn as sns
from learntools.advanced_pandas.indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q14(reviews.loc[reviews.country.isin(["France", "Italy"]) & (reviews.points >= 90), 'country'])
reviews['description']
reviews.description[0]
reviews.iloc[0]
reviews.iloc[:10]['description']
reviews.iloc[[1,2,3,5,8]]
reviews.loc[[0, 1, 10, 100], ['country', 'province', 'region_1', 'region_2']]
reviews.loc[0:999, ['country', 'variety']]
reviews.loc[reviews.country == 'Italy']
reviews.loc[reviews.region_2.notnull()]
ax = sns.countplot(x="points", data=reviews)
sns.countplot(x="points", data=reviews[:1000])
sns.countplot(x="points", data=reviews.iloc[-1000:])
sns.countplot(x='points', data=reviews[reviews.country == "Italy"])
reviews.loc[reviews.country.isin(["France", "Italy"]) & (reviews.points >= 90)].country