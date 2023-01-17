import pandas as pd

print("Setup Complete.")
fruit_sales = pd.DataFrame([[35, 21],[41, 34]], index=["2017 Sales", "2018 Sales"], columns=["Apples", "Bananas"])

print(fruit_sales)
ingredients = pd.Series(["4 cups", "1 cup", "2 large", "1 can"], 

                        index=["Flour", "Milk", "Eggs", "Spam"],

                        name = "Dinner")

print(ingredients)
reviews = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv", index_col=0)

print(reviews.head())
desc = reviews["description"]

print(desc)

# or

desc = reviews.description

print(desc)
first_description = reviews.description.iloc[0]

print(first_description)
indices = [0, 1, 10, 100]

labels = ["country", "province", "region_1", "region_2"]

df = reviews.loc[indices, labels]

print(df.head())
italian_wines = reviews[reviews.country == "Italy"]

print(italian_wines)
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
median_points = reviews.points.median()

print(median_points)

mean_points = reviews.points.mean()

print(mean_points)
countries = reviews.country.unique()

print(countries)
reviews_per_country = reviews["country"].value_counts()

print(reviews_per_country)
bargain_idx = (reviews.points / reviews.price).idxmax()

bargain_wine = reviews.loc[bargain_idx, 'title']

print(bargain_wine)
n_trop = reviews.description.map(lambda desc: "tropical" in desc).sum()

n_fruity = reviews.description.map(lambda desc: "fruity" in desc).sum()

descriptor_counts = pd.Series([n_trop, n_fruity], index=['tropical', 'fruity'])

print(descriptor_counts)
def toStar(row):

    if row.points >= 95:

        return 3

    elif row.points >= 85:

        return 2

    else:

        return 1



star_ratings = reviews.apply(toStar,axis="columns")

print(star_ratings)
reviews_written = reviews.groupby('taster_twitter_handle').size()

print(reviews_written)
best_rating_per_price = reviews.groupby('price').points.max().sort_index()

print(best_rating_per_price)
price_extremes = reviews.groupby('variety').price.agg([min,max])

print(price_extremes)
sorted_varieties = price_extremes.sort_values(by=['min', 'max'], ascending=False)

print(sorted_varieties)
reviewer_mean_ratings = reviews.groupby('taster_name').points.mean()

print(reviewer_mean_ratings)

reviewer_mean_ratings.describe()
country_variety_counts = reviews.groupby(["country", "variety"]).size().sort_values(ascending=False)

print(country_variety_counts)
dtype = reviews.points.dtype

print(dtype)
point_strings = reviews.points.astype("str")

print(point_strings)
n_missing_prices = len(reviews[reviews.price.isnull()])

print(n_missing_prices)
reviews_per_region = reviews.price.fillna("Unknown")

print(reviews_per_region)
renamed = reviews.rename(columns={"region_1": "region", "region_2": "locale"})

print(renamed)
reindexed = reviews.rename_axis("wines", axis="rows")

print(reindexed)