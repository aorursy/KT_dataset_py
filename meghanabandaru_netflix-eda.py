import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import collections
import re
import matplotlib.pyplot as plt
%matplotlib inline
netflix_data = pd.read_csv("../input/netflix-shows/netflix_titles.csv")
netflix_data.head()
netflix_data.info()
netflix_data.drop("show_id", axis=1, inplace=True)
netflix_data.head()
# Vertical bar chart

sns.countplot(x="type", data=netflix_data, palette="Blues_d")
# Horizontal bar chart

sns.countplot(y="type", data=netflix_data, palette="Blues_d")
# Number of movies/shows released by a director
sns.countplot(x="director", data= netflix_data)
# number of unique actors
netflix_data["cast"] = netflix_data["cast"].str.split(",")
netflix_data["cast"].explode().nunique()
netflix_data["cast"].explode()
# number of nulls
netflix_data.country.isna().sum()
country_data = netflix_data[netflix_data['country'].notna()]
country_data.country.isna().sum()
# data["Team"]= data["Team"].str.split("t", n = 1, expand = True)
country_data["country"] = country_data["country"].str.split(",")
country_data.country.explode().nunique()
country_data.country.explode()
country_data.country.explode().value_counts()
# Remove the spaces
countries = country_data.country.explode()
countries = [country.strip() for country in countries]
counter = collections.Counter(countries)
# print(counter)
print(counter.most_common(5))
top_countries = counter.most_common(5)
type(top_countries)
# Visualize top countries
top_countries_df = pd.DataFrame(top_countries, columns=['country','count'])
top_countries_df
sns.barplot(x="country", y="count", data=top_countries_df)
# Create a month column
netflix_data["month"] = pd.DatetimeIndex(netflix_data["date_added"]).month_name()
netflix_data.head()
plot = sns.countplot(x="month", data=netflix_data)
plot.set_xticklabels(plot.get_xticklabels(), rotation=40,  ha="right")
print(netflix_data.release_year.min())
print(netflix_data.release_year.max())
netflix_data.release_year.nunique()
type(netflix_data.release_year.sort_values())
def find_missing_years(years):
    return [x for x in range(years[0], years[-1]+1) if x not in years]
years = netflix_data.release_year.sort_values().tolist()
missing_years = find_missing_years(years)
missing_years
netflix_data[netflix_data["release_year"].isin(missing_years)]
netflix_data["release_year"][0]
decades = {
    "1960-1970":np.arange(1960, 1970,1),
    "1970-1980":np.arange(1970, 1980, 1),
    "1980-1990":np.arange(1980, 1990, 1),
    "1990-2000":np.arange(1990, 2000, 1),
    "2000-2010":np.arange(2000, 2010, 1),
    "2010-2020":np.arange(2010, 2020, 1)
}
decades
netflix_data.release_year[0]
year = 2019
for d,y in decades.items():
    if year in y:
        print(d)
for year in netflix_data.release_year:
    for d,y in decades.items():
        if year in y:
            netflix_data.loc[netflix_data["release_year"] == year, "decade"] = d
netflix_data.head()
netflix_data.decade.unique()
plot = sns.countplot(x="decade", data=netflix_data)
plot.set_xticklabels(plot.get_xticklabels(), rotation=40,  ha="right")
decade_df = netflix_data[netflix_data["decade"].notna()]
decade_df.decade.isna().sum()
plt.hist(decade_df.decade, density=True, bins=5)  # `density=False` would make counts
plt.ylabel('count')
plt.xlabel('decade');
# Which year has highest content
year_counter = collections.Counter(netflix_data.release_year)
year_counter.most_common(5)
highest_content_years_df = pd.DataFrame(year_counter.most_common(5), columns=['year','count'])
highest_content_years_df
# Visualize the highest years
sns.barplot(x="year", y="count", data=highest_content_years_df)
sns.lineplot(x=netflix_data.release_year.value_counts().index, y=netflix_data.release_year.value_counts())
netflix_data.rating.nunique()
netflix_data.rating.unique()
plot = sns.barplot(x=netflix_data.rating.value_counts().index, y=netflix_data.rating.value_counts())
plot.set_xticklabels(plot.get_xticklabels(), rotation=40,  ha="right")
netflix_data.duration.unique
# Check for nulls
netflix_data.listed_in.isna().sum()
netflix_data["listed_in"] = netflix_data["listed_in"].str.split(",")
netflix_data.listed_in.explode().nunique()
netflix_data.listed_in.explode().unique()
# Remove the spaces
categories = netflix_data.listed_in.explode()
categories = [category.strip() for category in categories]
cat_counter = collections.Counter(categories)
# print(counter)
print(cat_counter.most_common(5))
len(set(categories))
categories_df = pd.DataFrame(cat_counter.most_common(5), columns=['category','count'])
categories_df
plot = sns.barplot(x="category", y ="count", data=categories_df)
plot.set_xticklabels(plot.get_xticklabels(), rotation=40,  ha="right")
netflix_data.description[0]
netflix_data.drop("description", axis=1, inplace=True)
netflix_data.head()
sns.set(style="whitegrid")
sns.boxplot(x="release_year", y="type", data=netflix_data, palette="Set3")
yearly_type_data = netflix_data[netflix_data["decade"] == "2010-2020"]
yearly_type_data.head()
sns.set(style="whitegrid")
sns.boxplot(x="release_year", y="type", data=yearly_type_data, palette="Set3")
for index, (title, content_type, duration) in enumerate(zip(netflix_data.title, netflix_data.type, netflix_data.duration)):
    if content_type=="Movie":
        netflix_data.loc[netflix_data["title"] == title, "multiplier"] = 1
    else:
        num_of_seasons = re.findall(r'\d+',duration)
        netflix_data.loc[netflix_data["title"] == title, "multiplier"] = num_of_seasons
netflix_data.tail()
sns.set(style="whitegrid")
sns.boxplot(x="release_year", y="type", data=yearly_type_data, palette="Set3")
netflix_data.multiplier = pd.to_numeric(netflix_data['multiplier'])
sample = netflix_data.groupby(["release_year", "type"]).agg({'multiplier': 'sum'}).reset_index()
sample
# test for a random observation if the groupby is correct
sample = sample[sample["release_year"] >= 2010]
sample
g = sns.catplot(x="release_year", y="multiplier", hue="type", data=sample,
                height=6, kind="bar", palette="muted")
g.despine(left=True)
duration = country_data.explode("country")
duration.head()
# delete nulls in country
duration = duration[duration["country"].notna()]
duration.country.isna().sum()
# Filter out only movies that are released after 2010
duration = duration[(duration["type"] == "Movie") & (duration["release_year"] >= 2010)]
duration.type.unique()
top_countries = ["United States", "India", "United Kingdom", "Canada", "France"]
# Select only top countries
duration = duration.query("country in @top_countries")
duration.country.unique()
duration["duration"].replace({"min": ""}, inplace=True, regex=True)
duration.head()
#convert the duration col to int
duration["duration"] = duration["duration"].astype(int)
#Get the average duration for each country
duration = duration.groupby('country', as_index=False)['duration'].mean()
duration
sns.barplot(x="country", y="duration", data=duration)
sns.lineplot(x="country", y="duration", data=duration)
grouped_directors =  netflix_data.groupby(["director","type"]).size().nlargest(15).reset_index()
grouped_directors
grouped_directors =  netflix_data.groupby(["director","type"]).size().reset_index()
TV_dirs = grouped_directors[grouped_directors["type"]=="TV Show"]
movie_dirs = grouped_directors[grouped_directors["type"]=="Movie"]
pd.merge(TV_dirs, movie_dirs, on='director')
# Unstack the ```country``` into multiple rows for each country
countries = country_data.explode("country")
countries
top_countries
countries = countries.query("country in @top_countries")
countries
countries.country.unique()
countries = countries.groupby(["country", "type"]).size().reset_index()
countries
countries.columns=["country", "type", "count"]
countries
g = sns.catplot(x="country", y="count", hue="type", data=countries,
                height=6, kind="bar", palette="muted")
g.despine(left=True)
