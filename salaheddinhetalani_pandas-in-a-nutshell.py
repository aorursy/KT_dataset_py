import pandas as pd
pd.DataFrame({'Bob': ['I liked it.', 'It was awful.'], 'Sue': ['Pretty good.', 'Bland.']})
reviews = pd.DataFrame({'Bob': ['I liked it.', 'It was awful.'], 'Sue': ['Pretty good.', 'Bland.']}, index=['Product A', 'Product B'])

reviews
reviews.rename_axis("Products", axis='rows').rename_axis("Reviewers", axis='columns')
pd.Series([30, 35, 40], index=['2015 Sales', '2016 Sales', '2017 Sales'], name='Product A')
reviews.to_csv("reviews.csv")
wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv")

wine_reviews.head()
wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

wine_reviews.head()
wine_reviews = wine_reviews.rename(columns={'region_1': 'region', 'region_2': 'locale'})

wine_reviews
print(wine_reviews.country)

wine_reviews.country[0]
print(wine_reviews['country'])

wine_reviews['country'][:5]
wine_reviews.head()
wine_reviews.iloc[0]
wine_reviews.iloc[:3,0]
wine_reviews.iloc[[0, 1, 2], [0,1]]
wine_reviews.country.iloc[1]
wine_reviews.head()
wine_reviews.loc[0, 'country']
wine_reviews.loc[[0,1,10,100],['country', 'province', 'region', 'locale']]
wine_reviews.loc[[1, 2, 3, 5, 8]]
wine_reviews[(wine_reviews.country == 'Brazil')]
wine_reviews.loc[(wine_reviews.country.isin(['Australia', 'New Zealand'])) & (wine_reviews.points >= 95)]
wine_reviews.loc[wine_reviews.price.isnull()]
wine_reviews.set_index("title")
wine_reviews.reset_index(drop=True)
wine_reviews['critic'] = 'everyone'

wine_reviews['critic']
wine_reviews['index_backwards'] = range(len(wine_reviews), 0, -1)

wine_reviews
wine_reviews.points.describe()
wine_reviews.taster_name.describe()
wine_reviews.head()
wine_reviews.columns
wine_reviews.price.mean()
wine_reviews.price.median()
wine_reviews.taster_name.unique()
wine_reviews.nunique()
wine_reviews.taster_name.nunique()
wine_reviews.country.value_counts()
wine_reviews.count()
wine_reviews.price.min()
bargain_idx = (wine_reviews.points / wine_reviews.price).idxmax()

wine_reviews.loc[bargain_idx, 'title']
wine_reviews.points.dtype
wine_reviews.price.dtype
wine_reviews.country.dtype
wine_reviews.dtypes
wine_reviews.select_dtypes(include='object')
wine_reviews.select_dtypes(exclude='object')
wine_reviews.points.astype('float64')
wine_reviews.groupby('country').price.max()
wine_reviews.groupby('points').price.min()
wine_reviews.groupby(['country', 'province']).apply(lambda df: df.loc[df.points.idxmax()])
wine_reviews.groupby(['country']).points.agg([len, min, max])
countries_reviewed = wine_reviews.groupby(['country', 'province']).points.agg([max])

countries_reviewed = countries_reviewed.reset_index()

countries_reviewed.sort_values(by='max', ascending=False)
countries_reviewed.sort_values(by=['country', 'max'])
countries_reviewed.sort_index()
mean = wine_reviews.price.mean()

centered_price = wine_reviews.price.map(lambda p: p - mean)

centered_price
n_trop = wine_reviews.description.map(lambda desc: "tropical" in desc).sum()

n_fruity = wine_reviews.description.map(lambda desc: "fruity" in desc).sum()

descriptor_counts = pd.Series([n_trop, n_fruity], index=['tropical', 'fruity'])

descriptor_counts
def reduced_price(row):

    row.price = row.price - 1

    return row



new_price = wine_reviews.apply(reduced_price, axis='columns')

new_price
def starring(row):

    if row.country == 'Canada' or row.points >= 95:

        return 3

    elif row.points >= 85:

        return 2

    else:

        return 1

star_ratings = wine_reviews.apply(starring, axis='columns')

wine_reviews['rating'] = star_ratings

wine_reviews
wine_reviews['country - province'] = wine_reviews.country + " - " + wine_reviews.province

wine_reviews
wine_reviews.groupby(['country', 'province']).apply(lambda df: df.loc[df.points.idxmax()])
wine_reviews[wine_reviews.country.isnull()]
wine_reviews.locale.fillna("Unknown")
wine_reviews.taster_twitter_handle.replace("@kerinokeefe", "@kerino")
wine_reviews.drop(["index_backwards"], axis=1, inplace=True)

wine_reviews
wine_reviews.dropna(axis=0, subset=['price'], inplace=True)

wine_reviews
canadian_youtube = pd.read_csv("../input/youtube-new/CAvideos.csv")

british_youtube = pd.read_csv("../input/youtube-new/GBvideos.csv")
pd.concat([canadian_youtube, british_youtube])
canadian_youtube.set_index(['title', 'trending_date']).join(british_youtube.set_index(['title', 'trending_date']), lsuffix='_CAN', rsuffix='_UK')