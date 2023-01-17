# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import json
from pandas.io.json import json_normalize

style_data = json.load(open('../input/bjcp-beer-styles/styleguide-2015.json'))
categories = style_data['styleguide']['class'][0]['category']

# drill down into subcategories and append the parent category as a field on each subcategory
all_categories = []
for category in categories:
    for sub_category in category['subcategory']:
        sub_category['parent'] = category
        all_categories.append(sub_category)
        
styles = json_normalize(all_categories)
styles.head()
styles.dtypes
styles[['id', 'name', 'parent.id', 'parent.name']].sort_values('name').head()
beer_recipe = pd.read_csv('../input/beer-recipes/recipeData.csv', index_col='BeerID', encoding='latin1')
beer_recipe.head()
beer_recipe.dtypes
beer_styles = pd.read_csv('../input/beer-recipes/styleData.csv', encoding='latin1')
beer_styles.head()
joined_styles = styles \
    .sort_values('name') \
    .merge(beer_styles, how='outer', left_on='name', right_on='Style')
joined_styles[['id', 'name', 'parent.id', 'parent.name']].head()
joined_styles.isnull().sum()
joined_styles[joined_styles['Style'].isnull()][['id', 'name', 'parent.id', 'parent.name']].head(13)
def remove_historical_prefix(x):
    prefix = 'Historical Beer: '
    if x.startswith(prefix):
        return x[len(prefix):]
    return x

styles['name'] = styles['name'].apply(remove_historical_prefix)
joined_styles = styles \
    .sort_values('name') \
    .merge(beer_styles, how='outer', left_on='name', right_on='Style')

joined_styles[['id', 'name', 'parent.id', 'parent.name']].isnull().sum()
joined_styles[joined_styles['Style'].isnull()][['id', 'name', 'parent.id', 'parent.name']].head(13)
style_map = {
    'American Wheat or Rye Beer': 'Alternative Grain Beer',
    #'Belgian Specialty Ale': ???
    'Bohemian Pilsener': 'Czech Premium Pale Lager',
    'Brown Porter': 'Porter',
    'California Common Beer': 'California Common',
    'Classic American Pilsner': 'Pre-Prohibition Lager',
    'Classic Rauchbier': 'Rauchbier',
    'Dusseldorf Altbier': 'Altbier',
    #'Dark American Lager': ???
    'Dortmunder Export': 'German Helles Exportbier',
    'Dry Stout': 'Irish Stout',
    'Dunkelweizen': 'Dunkles Weissbier',
    'Extra Special/Strong Bitter (ESB)': 'Strong Bitter',
    # Brewer's Friend combined these two BJCP styles
    'Flanders Brown Ale/Oud Bruin': 'Oud Bruin',
    'German Pilsner (Pils)': 'German Pils',
    'Holiday/Winter Special Spiced Beer': 'Winter Seasonal Beer',
    #'Imperial IPA': ???
    #'Light American Lager': ???
    'Maibock/Helles Bock': 'Helles Bock',
    'Mild': 'Dark Mild',
    'North German Altbier': 'Altbier',
    'Northern English Brown': 'British Brown Ale',
    'Oktoberfest/Märzen': 'Festbier',
    #'Other Smoked Beer': ???
    #'Premium American Lager':
    #'Robust Porter': ???
    #'Roggenbier (German Rye Beer)': ???
    'Russian Imperial Stout': 'Imperial Stout',
    'Scottish Export 80/-': 'Scottish Export',
    'Scottish Heavy 70/-': 'Scottish Heavy',
    'Scottish Light 60/-': 'Scottish Light',
    'Southern English Brown': 'British Brown Ale',
    'Special/Best/Premium Bitter': 'Strong Bitter',
    #'Specialty Beer': ???
    #'Spice  Herb  or Vegetable Beer': ???
    #'Standard American Lager': ???
    'Standard/Ordinary Bitter': 'Ordinary Bitter',
    'Straight (Unblended) Lambic': 'Lambic',
    'Strong Scotch Ale': 'Wee Heavy',
    #'Traditional Bock':
    'Weizen/Weissbier': 'Weissbier'
}
def rename_styles(x):
    return style_map.get(x) or x

beer_styles['Style'] = beer_styles['Style'].apply(rename_styles)
joined_styles = styles \
    .sort_values('name') \
    .merge(beer_styles, how='outer', left_on='name', right_on='Style')

joined_styles.isnull().sum()
joined_styles[joined_styles['name'].isnull()][['Style']].head(31)
def get_sg_from_plato(plato):
    sg = 1 + (plato / (258.6 - ( (plato/258.2) *227.1) ) )
    return sg

beer_recipe['OG_sg'] = beer_recipe.apply(lambda row: get_sg_from_plato(row['OG']) if row['SugarScale'] == 'Plato' else row['OG'], axis=1)
beer_recipe['FG_sg'] = beer_recipe.apply(lambda row: get_sg_from_plato(row['FG']) if row['SugarScale'] == 'Plato' else row['FG'], axis=1)
beer_recipe['BoilGravity_sg'] = beer_recipe.apply(lambda row: get_sg_from_plato(row['BoilGravity']) if row['SugarScale'] == 'Plato' else row['BoilGravity'], axis=1)
import seaborn as sns
sns.set(style="whitegrid")

ax = sns.boxplot(data=beer_recipe[['OG_sg', 'FG_sg', 'BoilGravity_sg']], orient='h')
def clean_sg(sg):
    if sg > 1.5:
        sg = 1 + (sg / (258.6 - ( (sg/258.2) *227.1) ) )
        return sg
    else:
        return sg

beer_recipe['OG_sg'] = beer_recipe.apply(lambda row: clean_sg(row['OG_sg']), axis=1)
beer_recipe['FG_sg'] = beer_recipe.apply(lambda row: clean_sg(row['FG_sg']), axis=1)
beer_recipe['BoilGravity_sg'] = beer_recipe.apply(lambda row: clean_sg(row['BoilGravity_sg']), axis=1)
ax = sns.boxplot(data=beer_recipe[['OG_sg', 'FG_sg', 'BoilGravity_sg']], orient='h')
len(beer_recipe[(beer_recipe.OG_sg < beer_recipe.FG_sg)])
joined_recipe = beer_recipe \
                    .merge(joined_styles, how='inner', left_on='StyleID', right_on='StyleID')
joined_recipe.head()
def is_outlier(row, test_col, min_col, max_col):
    x = row[test_col]
    min_x = float(row[min_col])
    max_x = float(row[max_col])
    if x < min_x:
        return 'Low'
    elif x > max_x:
        return 'High'
    else:
        return 'InRange'

# drop recipies without OG, FG, IBU, Color
joined_recipe = joined_recipe[joined_recipe.StyleID.notnull() & \
                              joined_recipe.OG_sg.notnull() & \
                              joined_recipe.FG_sg.notnull() & \
                              joined_recipe.IBU.notnull() & \
                              joined_recipe.Color.notnull() & \
                              joined_recipe.ABV.notnull()]
joined_recipe['OG_class'] = joined_recipe.apply(lambda x: is_outlier(x, 'OG_sg', 'stats.og.low', 'stats.og.high'), axis=1)
joined_recipe['FG_class'] = joined_recipe.apply(lambda x: is_outlier(x, 'FG_sg', 'stats.fg.low', 'stats.fg.high'), axis=1)
joined_recipe['IBU_class'] = joined_recipe.apply(lambda x: is_outlier(x, 'IBU', 'stats.ibu.low', 'stats.ibu.high'), axis=1)
joined_recipe['SRM_class'] = joined_recipe.apply(lambda x: is_outlier(x, 'Color', 'stats.srm.low', 'stats.srm.high'), axis=1)
joined_recipe['ABV_class'] = joined_recipe.apply(lambda x: is_outlier(x, 'ABV', 'stats.abv.low', 'stats.abv.high'), axis=1)
joined_recipe.head()
import matplotlib
top_style_ids = joined_recipe['StyleID'].value_counts()[:20].index

fig, ax = matplotlib.pyplot.subplots(figsize=(20,6))
fplt = sns.swarmplot(x="name", y="OG_sg", hue="OG_class", hue_order=["Low", "InRange", "High"],
                     data=joined_recipe[(joined_recipe.StyleID.isin(top_style_ids))].sample(2000),
                     ax=ax)
fplt.axes.xaxis.set_tick_params(rotation=90)
fplt.set(ylim=(1.0, 1.2))
top_style_ids = joined_recipe['StyleID'].value_counts()[:20].index

fig, ax = matplotlib.pyplot.subplots(figsize=(20,6))
fplt = sns.swarmplot(x="name", y="FG_sg", hue="FG_class", hue_order=["Low", "InRange", "High"],
                     data=joined_recipe[(joined_recipe.StyleID.isin(top_style_ids))].sample(2000),
                     ax=ax)
fplt.axes.xaxis.set_tick_params(rotation=90)
top_style_ids = joined_recipe['StyleID'].value_counts()[:20].index

fig, ax = matplotlib.pyplot.subplots(figsize=(20,6))
fplt = sns.swarmplot(x="name", y="IBU", hue="IBU_class", hue_order=["Low", "InRange", "High"],
                     data=joined_recipe[(joined_recipe.StyleID.isin(top_style_ids))][(joined_recipe.IBU.between(0, 250))].sample(2000),
                     ax=ax)
fplt.axes.xaxis.set_tick_params(rotation=90)
top_style_ids = joined_recipe['StyleID'].value_counts()[:20].index

fig, ax = matplotlib.pyplot.subplots(figsize=(20,6))
fplt = sns.swarmplot(x="name", y="Color", hue="SRM_class", hue_order=["Low", "InRange", "High"],
                     data=joined_recipe[(joined_recipe.StyleID.isin(top_style_ids))].sample(2000),
                     ax=ax)
fplt.axes.xaxis.set_tick_params(rotation=90)
top_style_ids = joined_recipe['StyleID'].value_counts()[:20].index

fig, ax = matplotlib.pyplot.subplots(figsize=(20,6))
fplt = sns.swarmplot(x="name", y="ABV", hue="ABV_class", hue_order=["Low", "InRange", "High"],
                     data=joined_recipe[(joined_recipe.StyleID.isin(top_style_ids))][(joined_recipe.ABV.between(0, 20))].sample(2000),
                     ax=ax)
fplt.axes.xaxis.set_tick_params(rotation=90)
import matplotlib.pyplot as plt

grid = sns.FacetGrid(data=joined_recipe[(joined_recipe.StyleID.isin(top_style_ids))].sample(2000),
                     col="name", hue="name", col_wrap=5, size=3)
grid.map(plt.scatter, "IBU", "OG_sg", edgecolor="w")
grid.set(xlim=(5, 65), ylim=(1.025, 1.1))
#grid.axes[0].lines.extend([l1])
for ax in grid.axes:
    slopes = [(1.1 - 1.025) / (28 - 5),
              (1.1 - 1.025) / (35 - 7),
              (1.1 - 1.025) / (44 - 11),
              (1.1 - 1.025) / (53 - 14),
              (1.1 - 1.025) / (63 - 17),
              (1.077 - 1.025) / (65 - 23)]
    intercept = 1.0
    for slope in slopes:
        x_vals = np.array(ax.get_xlim())
        y_vals = intercept + slope * x_vals
        ax.plot(x_vals, y_vals, '--')
    name = ax.title._text.split('=')[1].strip()
    ibu_min = float(joined_styles[joined_styles.name == name].iloc[0]['stats.ibu.low'])
    ibu_max = float(joined_styles[joined_styles.name == name].iloc[0]['stats.ibu.high'])
    og_min = float(joined_styles[joined_styles.name == name].iloc[0]['stats.og.low'])
    og_max = float(joined_styles[joined_styles.name == name].iloc[0]['stats.og.high'])
    ax.axvspan(ibu_min, ibu_max, facecolor='0.5', alpha=0.25)
    ax.axhspan(og_min, og_max, facecolor='0.5', alpha=0.25)
grid = sns.FacetGrid(data=joined_recipe[(joined_recipe.StyleID.isin(top_style_ids))].sample(2000),
                     col="name", hue="name", col_wrap=5, size=3)
grid.map(plt.scatter, "ABV", "Color", edgecolor="w")
grid.set(xlim=(0, 15), ylim=(0, 60))
#grid.axes[0].lines.extend([l1])
for ax in grid.axes:
    name = ax.title._text.split('=')[1].strip()
    abv_min = float(joined_styles[joined_styles.name == name].iloc[0]['stats.abv.low'])
    abv_max = float(joined_styles[joined_styles.name == name].iloc[0]['stats.abv.high'])
    srm_min = float(joined_styles[joined_styles.name == name].iloc[0]['stats.srm.low'])
    srm_max = float(joined_styles[joined_styles.name == name].iloc[0]['stats.srm.high'])
    ax.axvspan(abv_min, abv_max, facecolor='0.5', alpha=0.25)
    ax.axhspan(srm_min, srm_max, facecolor='0.5', alpha=0.25)
mega_bitter = joined_recipe[(joined_recipe.IBU > 200)]
len(mega_bitter)
# This is used for fast string concatination
from io import StringIO
si=StringIO()
mega_bitter['Name'].apply(lambda x: si.write(str(x) + ' '))
s=si.getvalue()
si.close()
# Note sure how meaningful this is
# but here's a look.
s[0:400]
from wordcloud import WordCloud

# Read the whole text.
text = s

# Generate a word cloud image
wordcloud = WordCloud().generate(text)

# Display the generated image:
# the matplotlib way:
import matplotlib.pyplot as plt


# take relative word frequencies into account, lower max_font_size
wordcloud = WordCloud(background_color="white",
                      max_words=len(s),
                      width=1600, height=800,
                      relative_scaling=.5).generate(text)
plt.figure(figsize=(15,7))
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()
# Though the following import is not directly being used, it is required
# for 3D projection to work
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans
from sklearn import datasets

american_ipa = joined_recipe[(joined_recipe.name == 'American IPA')]
len(american_ipa)

X = american_ipa[(american_ipa.ABV < 10)][['ABV', 'Color']]

est = KMeans(n_clusters=2)
est.fit(X)
labels = est.labels_
X['cluster'] = labels
X['Name'] = american_ipa['Name']

X.head()
g = sns.lmplot(x="ABV", y="Color", hue="cluster",
               truncate=True, size=8, fit_reg=False, data=X)
si=StringIO()
X[(X.cluster == 1)]['Name'].apply(lambda x: si.write(str(x) + ' '))
s=si.getvalue()
si.close()
# Note sure how meaningful this is
# but here's a look.
s[0:400]
# Read the whole text.
text = s

# Generate a word cloud image
wordcloud = WordCloud().generate(text)

# Display the generated image:
# the matplotlib way:
import matplotlib.pyplot as plt


# take relative word frequencies into account, lower max_font_size
wordcloud = WordCloud(background_color="white",
                      max_words=len(s),
                      width=1600, height=800,
                      relative_scaling=.5).generate(text)
plt.figure(figsize=(15,7))
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()
american_light_lager = joined_recipe[(joined_recipe.name == 'American Light Lager')]
len(american_light_lager)
def in_style(recipes, style):
    num_recipes = len(recipes)
    og_low = float(style['stats.og.low'])
    og_high = float(style['stats.og.high'])
    fg_low = float(style['stats.fg.low'])
    fg_high = float(style['stats.fg.high'])
    ibu_low = float(style['stats.ibu.low'])
    ibu_high = float(style['stats.ibu.high'])
    srm_low = float(style['stats.srm.low'])
    srm_high = float(style['stats.srm.high'])
    matches = recipes[(recipes.OG_sg.between(og_low, og_high))] \
                     [(recipes.FG_sg.between(fg_low, fg_high))] \
                     [(recipes.IBU.between(ibu_low, ibu_high))] \
                     [(recipes.Color.between(srm_low, srm_high))]
    num_matches = len(matches)
    return num_matches / num_recipes

recipe_matches = {}

for style_name in joined_styles['Style'].unique():
    try:
        style = joined_styles[(joined_styles.Style == style_name)].iloc[0]
    except:
        pass
    recipe_matches[style.Style] = in_style(american_light_lager, style)
    
matches_df = pd.DataFrame({'name': list(recipe_matches.keys()), 'match': list(recipe_matches.values())})
matches_df.head()
matches_df.sort_values('match', ascending=False).head(20)