import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import pprint, math

import matplotlib.pyplot as plt



# Read in the data files



# For the part number:

inv_parts_data = pd.read_csv("../input/lego-database/inventory_parts.csv")



# Sets and the year they were released

sets_data = pd.read_csv("../input/lego-database/sets.csv")



# To assist the join between part-number, and year introduced 

inv_data = pd.read_csv("../input/lego-database/inventories.csv")



# We'll need this to track parent themes

theme_data = pd.read_csv("../input/lego-database/themes.csv")



# We want to know the type of part

part_cat_data = pd.read_csv("../input/lego-database/parts.csv")



# Join data into a table mapping part-number (with duplicates) to year

years_set_data = pd.merge(inv_data, sets_data,on='set_num')

years = years_set_data[['id', 'set_num','year', 'theme_id']]

data = pd.merge(inv_parts_data, years, left_on='inventory_id', right_on='id')

data = pd.merge(part_cat_data, data, on='part_num')

data = data.drop(['id', 'quantity', 'is_spare', 'name'], axis='columns').sort_values(by=['year'])



# Clean up parts that appear in duplicate rows for a single set (each color is listed separately)

data = data.drop_duplicates(subset=['part_num', 'set_num'], keep='first')



# Add the parent theme



memo = {}

def get_parent_theme(theme):

    if theme in memo:

        return memo[theme]

    

    parent = theme_data.loc[theme_data['id'] == theme]['parent_id']

    if math.isnan(parent):

        return theme

    else:

        parent = int(parent)

    if parent == theme:

        r = theme

    else:

        r = get_parent_theme(parent)

    

    memo[theme] = r

    return r



data['parent_theme_id'] = data['theme_id'].apply(lambda x: get_parent_theme(x) )



print(data.head())

# Find the year each part was introduced



part_sum = {} # part_num : (sets_containing, year_introduced)

for _, inv in data.iterrows():

    part_num = inv['part_num']



    if part_num in part_sum:

        s = part_sum[part_num][0] + 1

        yr = min(part_sum[part_num][1], inv['year'])

    else:

        s = 1

        yr = inv['year']

    part_sum[part_num] = (s, yr)





# Find number of sets released each year



set_releases = {}

for _, row in sets_data.iterrows():

    yr = row['year']

    if yr in set_releases:

        set_releases[yr] +=1

    else:

        set_releases[yr] = 1



#print("New sets released by year:")

#pprint.pprint(set_releases)





        
# Parts introduced each year

year_part_count = {}



# For each part, number of sets that it appears in, summed per year (ie. sum of set-appearances for all parts originating in year)

part_appearances_by_year_first_seen = {}



for part_num in part_sum:

    yr = part_sum[part_num][1]

    s = part_sum[part_num][0]

    

    if yr in year_part_count:

        year_part_count[yr] += 1

        part_appearances_by_year_first_seen[yr] += s

    else:

        year_part_count[yr] = 1

        part_appearances_by_year_first_seen[yr] = s



years = []

part_appearances = []

for yr in sorted(part_appearances_by_year_first_seen):

    years.append(yr)

    part_appearances.append(part_appearances_by_year_first_seen[yr])



first_seen = []

new_parts = []

new_parts_per_set = []

new_sets = []

sets_per_new_part = []

year_reuses = {}

for yr in year_part_count:

    if yr == 2017:

        continue # Seems data was collected mid-year

    first_seen.append(yr)

    new_parts.append(year_part_count[yr])

    new_sets.append(set_releases[yr])

    sets_per_new_part.append(part_appearances_by_year_first_seen[yr] / year_part_count[yr])

    year_reuses[yr] = part_appearances_by_year_first_seen[yr] / year_part_count[yr]

    new_parts_per_set.append(year_part_count[yr] / set_releases[yr])



first_seen_parts = pd.DataFrame.from_dict({'year': first_seen, 'new_parts': new_parts, 'new_sets': new_sets, 'sets_per_new_part': sets_per_new_part, 'new_parts_per_set': new_parts_per_set}).sort_values(by=['year'])



print(first_seen_parts.head())
#fig, ax = plt.subplots()

#first_seen_parts.plot.scatter(x='year',y='new_parts_per_set', c='new_sets', colormap='Wistia', ax=ax);



print("Zoom in on last 35 years")

#fig, ax = plt.subplots()

first_seen_parts[-35:].plot.bar(x='year',y='new_parts_per_set');#, c='new_sets', colormap='Wistia', ax=ax);
#All time

#fig, ax = plt.subplots()

#first_seen_parts.plot.scatter(x='year',y='sets_per_new_part', c='new_sets', colormap='Wistia', ax=ax);



print("Zoom in on last 35 years")

fig, ax = plt.subplots()

#first_seen_parts[-35:].plot.scatter(x='year',y='sets_per_new_part', c='new_sets', colormap='Wistia', ax=ax);

sns.regplot(x=first_seen_parts[-35:]['year'], y=first_seen_parts[-35:]['sets_per_new_part'])
all_sets = sum(new_sets)

t = 0

total_sets = []



for x in new_sets:

    t += x

    total_sets.append(all_sets - t)



first_seen_parts['sets_to_be_released'] = total_sets

first_seen_parts['percent_of_sets_containing_part'] = first_seen_parts['sets_per_new_part'] / first_seen_parts['sets_to_be_released'] * 100



#print(first_seen_parts)

fig, ax = plt.subplots()



print("Decline in reuse of new bricks")

print("Zoom in on last 35 years")

first_seen_parts[-35:].plot.scatter(x='year',y='percent_of_sets_containing_part', c='sets_per_new_part', colormap='Wistia', ax=ax);

# For each part, let's count how many themes it appeared in



themes_for_part = {} # part_num : list of themes

parent_themes_for_part = {} # part_num : list of parent themes

for _, inv in data.iterrows():

    part_num = inv['part_num']

    theme = inv['theme_id']

    parent_theme = get_parent_theme(theme)

    if part_num not in themes_for_part:

        themes_for_part[part_num] = [theme]

        parent_themes_for_part[part_num] = [parent_theme]

    else:

        if theme not in themes_for_part[part_num]:

            themes_for_part[part_num].append(theme)

        if parent_theme not in parent_themes_for_part[part_num]:

            parent_themes_for_part[part_num].append(parent_theme)





themes_for_year = {} # year: themes * parts

parent_themes_for_year = {} # year: themes * parts

for part_num in part_sum:

        yr = part_sum[part_num][1]

        themes = len(themes_for_part[part_num])

        parent_themes = len(parent_themes_for_part[part_num])

        if yr in themes_for_year:

            themes_for_year[yr] += themes

            parent_themes_for_year[yr] += parent_themes

        else:

            themes_for_year[yr] = themes

            parent_themes_for_year[yr] = parent_themes

years = []

theme_count = []

parent_theme_count = []

for yr in sorted(themes_for_year):

    if yr == 2017:

        continue

    years.append(yr)

    theme_count.append(themes_for_year[yr])

    parent_theme_count.append(parent_themes_for_year[yr])



first_seen_parts['themes_per_part'] = theme_count / first_seen_parts['new_parts']

first_seen_parts['parent_themes_per_part'] = parent_theme_count / first_seen_parts['new_parts']
#print(first_seen_parts.head())



# Plot parent themes - very similar results to themes

#sns.regplot(x=first_seen_parts[-35:]['year'], y=first_seen_parts[-35:]['parent_themes_per_part'])



sns.regplot(x=first_seen_parts[-35:]['year'], y=first_seen_parts[-35:]['themes_per_part'])
# An Aside: how many themes per parent (root) theme?

themes_per_parent = {}

for t in memo:

    p = memo[t]

    if p in themes_per_parent:

        themes_per_parent[p] +=1

    else:

        themes_per_parent[p] = 1



pparents = []

pcount = []

for p in sorted(themes_per_parent):

    pparents.append(p)

    pcount.append(themes_per_parent[p])



df = pd.DataFrame.from_dict({'parent_theme': pparents, 'themes_count': pcount})

df.plot.bar(x='parent_theme',y='themes_count');



# Let's try to make some predictions



import numpy as np

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

from sklearn.utils import shuffle

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import OneHotEncoder







# We want to make a prediction based only on the information that's available at part origination

part_data = data.copy()

part_data = part_data[part_data.year < 2017 ]



part_data = part_data.drop(['inventory_id', 'set_num'], axis='columns').sort_values(by=['part_num'])

part_data = part_data.drop_duplicates(subset=['part_num'], keep='first')

    

part_data['reuse_for_year'] = part_data['year'].apply(lambda yr: year_reuses[yr] )



part_data['uses'] = part_data['part_num'].apply(lambda x: part_sum[x][0] )



# We'll class pieces with over 30 reuses as "high use" and bunch them together

part_data['uses'] = part_data['uses'].clip(1, 31)



part_data.reset_index(drop=True, inplace=True)

print(part_data.head())



# Convert part categories, colours, themes and parent themes to One Hot Encoding



def convert_to_one_hot(df, key, enc):

    enc_df = pd.DataFrame(enc.fit_transform(df[[key]]).toarray())

    df = pd.concat([df, enc_df], axis=1)

    df = df.drop([key], axis='columns')

    return df





enc = OneHotEncoder()

for k in ['part_cat_id', 'color_id', 'theme_id', 'parent_theme_id']:

    part_data = convert_to_one_hot(part_data, k, enc)



# Randomize row order before splitting

part_data = shuffle(part_data, random_state=3)



y = part_data['uses']



part_data = part_data.drop(['uses', 'part_num'], axis='columns')

X = part_data

# Split into validation and training data

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=2)



print("%s rows of training data"%(len(train_y)))

print("%s rows of validation data"%(len(val_y)))

# Define and fit the model.

rf_model = RandomForestRegressor(random_state=1)

rf_model.fit(train_X, train_y)



# Make predictions on the validation data

predictions = rf_model.predict(val_X)



# Calculate the error of the predictions

rf_val_mae = mean_absolute_error(val_y, predictions)

rf_val_rmse = mean_squared_error(val_y, predictions)



print("\n")

print("Validation RMSE: {}".format(rf_val_rmse))



print("\n\n** The Mean Average Error on the validation set is {}! **".format(rf_val_mae))