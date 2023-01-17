import pandas as pd
pd.set_option('max_rows', 5)
wine = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
ramen = pd.read_csv("../input/ramen-ratings/ramen-ratings.csv", index_col=0)
stars = ramen['Stars']
na_stars = stars.replace('Unrated', None).dropna()
float_stars = na_stars.astype('float64')
float_stars.head()
(ramen['Stars']
     .replace('Unrated', None)
     .dropna()
     .astype('float64')
     .head())
wine.head()
wine.assign(
    region_1=wine.apply(lambda srs: srs.region_1 if pd.notnull(srs.region_1) else srs.province, 
                        axis='columns')
)
def name_index(df):
    df.index.name = 'review_id'
    return df

wine.pipe(name_index)