import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import seaborn as sns
from ast import literal_eval

%matplotlib inline
veg_meat = ["#454d66", "#b7e778", "#1fab89"]
sns.set_palette(veg_meat)
recipes = pd.read_csv('/kaggle/input/food-com-recipes-and-user-interactions/RAW_recipes.csv')
interactions = pd.read_csv('/kaggle/input/food-com-recipes-and-user-interactions/RAW_interactions.csv')
recipes.head()
print(recipes.info())
recipes.describe()
recipes[['minutes', 'n_steps', 'n_ingredients']].hist()
interactions.head()
print(interactions.info())
interactions.describe()
interactions['rating'].hist()
from_year, to_year = '2008-01-01','2017-12-31'

recipes['submitted'] = pd.to_datetime(recipes['submitted'])
recipes['submitted'] = recipes['submitted'].apply(lambda x: x.tz_localize(None))
recipes_l0y = recipes.loc[recipes['submitted'].between(from_year, to_year, inclusive=False)]

interactions['date'] = pd.to_datetime(interactions['date'])
interactions['date'] = interactions['date'].apply(lambda x: x.tz_localize(None))
interactions_l0y = interactions.loc[interactions['date'].between(from_year, to_year, inclusive=False)]

print(recipes_l0y.shape)
print(interactions_l0y.shape)
sns.boxplot(x=recipes_l0y["minutes"])
# calculate the first quartile, third quartile and the interquartile range
Q1 = recipes_l0y['minutes'].quantile(0.25)
Q3 = recipes_l0y['minutes'].quantile(0.75)
IQR = Q3 - Q1

# calculate the maximum value and minimum values according to the Tukey rule
max_value = Q3 + 1.5 * IQR
min_value = Q1 - 1.5 * IQR

# filter the data for values that are greater than max_value or less than min_value
minutes_outliers = recipes_l0y[(recipes_l0y['minutes'] > max_value) | (recipes_l0y['minutes'] < min_value)]
minutes_outliers.sort_values('minutes')
# filter out recipes that take longer than 730 days as outliers
recipes_l0y = recipes_l0y.query('minutes < 1051200')
recipes_l0y['year'] = recipes_l0y['submitted'].dt.year
interactions_l0y['year'] = interactions_l0y['date'].dt.year
ratings_by_recipe = interactions_l0y.groupby(['recipe_id', 'year']).agg(
    rating_cnt = ('rating', 'count'),
    rating_avg = ('rating', 'mean'),
)
ratings_by_recipe.head()
recipes_and_ratings = recipes_l0y.merge(ratings_by_recipe, left_on='id', right_on='recipe_id')
recipes_and_ratings.head(2)
# convert the tags column to list format
recipes_and_ratings['tags'] = recipes_and_ratings['tags'].apply(lambda x: literal_eval(str(x)))
# add vegetarian and vegan boolean columns
recipes_and_ratings['vegetarian'] = ['vegetarian' in tag for tag in recipes_and_ratings['tags']]
recipes_and_ratings['vegan'] = ['vegan' in tag for tag in recipes_and_ratings['tags']]
recipes_and_ratings = recipes_and_ratings.drop(columns=['name', 'tags', 'nutrition', 'steps', 'description', 'ingredients'])
recipes_and_ratings.head(2)
#plot a venn diagram of vegetarian and vegan recipe counts
vegetarian_cnt = len(recipes_and_ratings.query('vegetarian == True'))
vegan_cnt = len(recipes_and_ratings.query('vegan == True'))
intersect_cnt = len(recipes_and_ratings.query('vegetarian == True and vegan == True'))

venn2(subsets = (vegetarian_cnt, vegan_cnt-intersect_cnt, intersect_cnt), set_labels = ('Vegetarian', 'Vegan'), set_colors=('#b7e778', '#031c16', '#031c16'), alpha = 1)
df = recipes_and_ratings.groupby(['year', 'vegetarian']).agg(
    recipe_cnt = ('id', 'count')
).reset_index()

plt.figure(figsize=(12,6))

ax = sns.lineplot(data=df, x='year', y='recipe_cnt', hue='vegetarian', linewidth=2.5)
ax.set(ylim=(0, None))
ax.set_title('Number of new recipes by year')
ax
df = recipes_and_ratings.groupby(['year']).agg(
    total_cnt = ('id', 'count'),
    vegetarian_cnt = ('vegetarian', 'sum'),
    vegan_cnt = ('vegan', 'sum'),
).reset_index()

df['vegetarian_pct'] = df['vegetarian_cnt'] / df['total_cnt'] * 100
df['vegan_pct'] = df['vegan_cnt'] / df['total_cnt'] * 100

plt.figure(figsize=(12,6))

ax = sns.lineplot(data=pd.melt(df[['year', 'vegetarian_pct', 'vegan_pct']], ['year']), x='year', y='value', palette=veg_meat[1:], hue='variable', linewidth=2.5)
ax.set(ylim=(0, 100))
ax.set_title('Percent of vegetarian recipes by year')
ax
ratings_by_recipe = interactions_l0y.groupby(['recipe_id', 'year']).agg(
    rating_cnt = ('rating', 'count'),
    rating_avg = ('rating', 'mean'),
).reset_index()
ratings_by_recipe = ratings_by_recipe.merge(recipes_and_ratings[['id', 'vegetarian', 'vegan']], left_on='recipe_id', right_on='id')

df = ratings_by_recipe.groupby(['year', 'vegetarian']).agg(
    rating_cnt = ('rating_cnt', 'sum'),
    rating_avg = ('rating_avg', 'mean'),
).reset_index()

plt.figure(figsize=(12,6))

ax = sns.lineplot(data=df, x='year', y='rating_cnt', hue='vegetarian', linewidth=2.5)
ax.set_title('Recipe ratings by year')
ax
interactions_by_recipe_and_year = interactions_l0y.reset_index().groupby(['recipe_id', 'year']).agg(
    rating_cnt = ('index', 'count'),
    rating_avg = ('rating', 'mean'),
).reset_index()

interactions_and_recipes = interactions_by_recipe_and_year[['recipe_id', 'year', 'rating_cnt', 'rating_avg']].merge(recipes_and_ratings[['id', 'vegetarian', 'vegan']], left_on='recipe_id', right_on='id')

interactions_and_recipes['vegetarian_rating_cnt'] = np.where(interactions_and_recipes['vegetarian'] == True, interactions_and_recipes['rating_cnt'], 0)
interactions_and_recipes['vegan_rating_cnt'] = np.where(interactions_and_recipes['vegan'] == True, interactions_and_recipes['rating_cnt'], 0)

df = interactions_and_recipes.groupby(['year']).agg(
    total_cnt = ('rating_cnt', 'sum'),
    vegetarian_cnt = ('vegetarian_rating_cnt', 'sum'),
    vegan_cnt = ('vegan_rating_cnt', 'sum'),
).reset_index()

df['vegetarian_pct'] = df['vegetarian_cnt'] / df['total_cnt'] * 100
df['vegan_pct'] = df['vegan_cnt'] / df['total_cnt'] * 100

plt.figure(figsize=(12,6))

ax = sns.lineplot(data=pd.melt(df[['year', 'vegetarian_pct', 'vegan_pct']], ['year']), x='year', y='value', palette=veg_meat[1:], hue='variable', linewidth=2.5)
ax.set(ylim=(0, 100))
ax.set_title('Percent of votes on vegetarian recipes by year')
ax
df = ratings_by_recipe.groupby(['year', 'vegetarian']).agg(
    rating_avg = ('rating_avg', 'mean')
).reset_index()

plt.figure(figsize=(12,6))

ax = sns.lineplot(data=df, x='year', y='rating_avg', hue='vegetarian', linewidth=2.5)
ax.set(ylim=(0, 5))
ax.set_title('Average recipe rating by year')
ax
recipes_and_cohorts = recipes_and_ratings.copy()
recipes_and_cohorts['submitted_year'] = recipes_and_cohorts['submitted'].apply(lambda x: x.strftime('%Y'))
# add cohort column — the year of the user's first recipe submission
recipes_and_cohorts.set_index('contributor_id', inplace=True)
recipes_and_cohorts['contributor_cohort'] = recipes_and_cohorts.groupby(level=0)['submitted'].min().apply(lambda x: x.strftime('%Y'))
recipes_and_cohorts.reset_index(inplace=True)
recipes_and_cohorts.head()
def add_cohort_periods(df):
    """
    Creates a `cohort_period` column, which is the Nth period based on the contributor's first recipe.
    """
    df['cohort_period'] = np.arange(len(df)) + 1
    return df

def group_into_cohorts(df):
    """
    Aggregates contributor count, recipe count and cohort period by contributor cohort and year of submission.
    """
    df = df.groupby(['contributor_cohort', 'submitted_year']).agg(
        contributor_cnt = ('contributor_id', 'nunique'),
        recipe_cnt = ('id', 'nunique'),
    )
    df = df.groupby('contributor_cohort').apply(add_cohort_periods)
    return df

# non-vegetarian cohorts
cohorts_nonveg = group_into_cohorts(recipes_and_cohorts[recipes_and_cohorts['vegetarian'] == False])

# vegetarian cohorts
cohorts_veg = group_into_cohorts(recipes_and_cohorts[recipes_and_cohorts['vegetarian'] == True])
cohorts_veg.head()
def calculate_cohort_sizes(df):
    """
    Calculates cohort sizes.
    """
    df.reset_index(inplace=True)
    df.set_index(['contributor_cohort', 'cohort_period'], inplace=True)
    return df['contributor_cnt'].groupby('contributor_cohort').first()

# calculate cohort sizes
cohort_sizes_nonveg = calculate_cohort_sizes(cohorts_nonveg)
cohort_sizes_veg = calculate_cohort_sizes(cohorts_veg)
cohort_sizes_veg.head()
def convert_cohort_counts_to_pct(df, cohort_sizes):
    """
   Converts cohort period contributor counts to percentages.
    """
    df = df.unstack(0).divide(cohort_sizes, axis=1)
    df.reset_index(inplace=True)
    return df

# convert cohort period contributor counts to percentages
contributor_retention_nonveg = convert_cohort_counts_to_pct(cohorts_nonveg['contributor_cnt'], cohort_sizes_nonveg)
contributor_retention_veg = convert_cohort_counts_to_pct(cohorts_veg['contributor_cnt'], cohort_sizes_veg)
contributor_retention_veg
def plot_retention_curves(df, cohorts, title, position):
    """
   Plots retention curves for cohorts.
    """
    plot = sns.lineplot(
        data=pd.melt(contributor_retention_nonveg[['cohort_period'] + cohorts], ['cohort_period']),
        x='cohort_period',
        y='value',
        palette='rocket_r',
        hue='contributor_cohort',
        linewidth=2.5,
        ax=ax[position])
    plot.set(xlim=(0, 8))
    plot.set(ylim=(0, 1))
    plot.set(xlabel='Cohort period')
    plot.set(ylabel='Active contributors')
    plot.set_title('Contributor retention by cohort: ' + title)
    return

# plot contributor retention curves
fig, ax = plt.subplots(1, 2, figsize=(12,6))

cohorts_to_display = ['2008', '2009', '2010', '2011']

plot_retention_curves(contributor_retention_nonveg, cohorts_to_display, 'Non-vegetarian', 0)
plot_retention_curves(contributor_retention_veg, cohorts_to_display, 'Vegetarian', 1)

fig.show()
# get first recipe by contributor
df = recipes_and_cohorts.groupby('contributor_id').agg(
    vegetarian = ('vegetarian', 'mean'),
    contributor_cohort = ('contributor_cohort', 'min'),
)
# counting contributors with >50% of vegetarian contibutions as vegetarians
df.reset_index(inplace=True)
df = df.round(0)

# get first recipe by contributor
df = df.groupby(['contributor_cohort', 'vegetarian']).agg(
    contributor_cnt = ('contributor_id', 'count'),
)
# counting contributors with >50% of vegetarian contibutions as vegetarians
df.reset_index(inplace=True)
df['vegetarian'] = df['vegetarian'].astype(bool)

plt.figure(figsize=(12,6))

ax = sns.lineplot(data=df, x='contributor_cohort', y='contributor_cnt', palette=veg_meat[:2], hue='vegetarian', linewidth=2.5)
ax.set(xlabel='New contributors')
ax.set(ylabel='Year')
ax.set_title('New contributors by year')
plt.figure(figsize=(12,6))

ax = sns.lineplot(data=df, x='contributor_cohort', y='contributor_cnt', palette=veg_meat[:2], hue='vegetarian', linewidth=2.5)
ax.set(yscale="log")
ax.set(xlabel='New contributors')
ax.set(ylabel='Year')
ax.set_title('New contributors by year (log)')