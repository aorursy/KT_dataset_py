import pandas as pd
import json
from pandas.io.json import json_normalize
# define json string
data = [{'state': 'Florida', 
         'shortname': 'FL',
         'info': {'governor': 'Rick Scott'},
         'counties': [{'name': 'Dade', 'population': 12345},
                      {'name': 'Broward', 'population': 40000},
                      {'name': 'Palm Beach', 'population': 60000}]},
        {'state': 'Ohio',
         'shortname': 'OH',
         'info': {'governor': 'John Kasich'},
         'counties': [{'name': 'Summit', 'population': 1234},
                      {'name': 'Cuyahoga', 'population': 1337}]}]
# use normalization to create tables from nested element
json_normalize(data, 'counties')
# further populate tables created from nested element
json_normalize(data, 'counties', ['state', 'shortname', ['info', 'governor']])
# load json as string
json.load((open('../input/world-bank-project/world_bank_projects_less.json')))
# load as Pandas dataframe
sample_json_df = pd.read_json('../input/world-bank-project/world_bank_projects_less.json')
sample_json_df
df = pd.read_json('../input/world-bank-project/world_bank_projects.json')
print(df.shape)
df.describe()

# also load in the raw json to wrangle the nested fields later
with open('../input/world-bank-project/world_bank_projects.json') as f:
    raw = json.load(f)

print(df.head())
# group by country and count distinct on id's
df.groupby('country_namecode').id.nunique().sort_values(ascending=False).head(10)
# seems like a one to many relationship between themes (name and code) and id
df_themes = json_normalize(raw, 'mjtheme_namecode', ['id'])
print(df_themes.head(10))
# some projects seem to have multiple theme code's
x = df_themes.groupby('id').code.nunique().sort_values(ascending=False).head(10)
print(x)
# so to find top themes we will account for this one to many relationship
print('Top 10 Major World Bank Project Themes:')
df_themes.name.value_counts().head(10)
# looks like [name] missing for some rows
# create lookup table from code to name
df_themes_name_to_code = df_themes.groupby('name').code.max().sort_values(ascending=False)
# drop the missing name rows
df_themes_name_to_code = df_themes_name_to_code[df_themes_name_to_code.index != '']
# convert to df
df_themes_name_to_code = pd.DataFrame(df_themes_name_to_code,columns=['code'])
# pull name into a column
df_themes_name_to_code['name_clean'] = df_themes_name_to_code.index
# set code to be the index
df_themes_code_to_name = df_themes_name_to_code.set_index(['code'])
print(df_themes_name_to_code)
# now merge on the name based on the code for the missing projects
print(df_themes.shape)
df_themes_clean = df_themes.merge(df_themes_code_to_name,how='outer',left_on=['code'],right_index=True)
print(df_themes_clean)
print(df_themes_clean.shape)
##Now get top 10 themes by name after filling in the missing names based on the theme code

# based on pre cleaned data
print('Top 10 Major World Bank Project Themes (Original):')
print(df_themes.name.value_counts().head(10))
print('--------------------------------------------------')

# based on cleaning we have done
print('Top 10 Major World Bank Project Themes (Cleaned):')
print(df_themes_clean.name_clean.value_counts().head(10))
print('--------------------------------------------------')
##Collapse cleaned data and merge back to original data
# get a list of cleaned theme names by id 
df_theme_names = pd.DataFrame(df_themes_clean.groupby('id').apply(lambda x: '|'.join(x['name_clean'])),columns=['theme_names'])
# get s list of theme codes by id
df_theme_codes = pd.DataFrame(df_themes.groupby('id').apply(lambda x: '|'.join(x['code'])),columns=['theme_codes'])
# now make an id level lookup table thsat can be used later to merge to original data
df_theme_lut = df_theme_names.merge(df_theme_codes,left_index=True,right_index=True)
# look at shape
print(df_theme_lut.shape)
# take a look
df_theme_lut.head()
#Now finally merge collapsed "|" seperated list of theme codes and names for each row in original data

# merge back out cleaned fields into the original data
df_cleaned = df.merge(df_theme_lut,how='right',left_on=['id'],right_index=True,)

# make sure shape has has not changed
print(df.shape)
print(df_cleaned.shape)
# we expect to just see the addition of 2 cols

# take a look at the fields to see the cleaning we have done
df_cleaned[['id','mjtheme_namecode','theme_names','theme_codes']].head(10)