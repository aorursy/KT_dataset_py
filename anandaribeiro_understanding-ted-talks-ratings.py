import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from collections import defaultdict
import datetime
from pandas.plotting import parallel_coordinates
from scipy.stats import pearsonr
import re

%matplotlib inline
ted_data = pd.read_csv('../input/ted_main.csv')
ted_data.keys()
ted_data.head()
def transform_date(date):
    date_info = datetime.date.fromtimestamp(date)
    return date_info
    
ted_data['film_date'] = ted_data['film_date'].apply(transform_date)
ted_data['published_date'] = ted_data['published_date'].apply(transform_date)

ted_data['duration'] = ted_data['duration']/60

ted_data.head()
ted_data.dtypes
pd.isnull(ted_data).sum()
print ("First row with missing value")
for index, row in ted_data.iterrows():
    if pd.isnull(row['speaker_occupation']):
        print (row)
        break
ted_data.fillna('Unknown', inplace = True)

pd.isnull(ted_data).sum()
print (ted_data['languages'][ted_data['languages'] == 0].count())
ted_data[ted_data['languages'] == 0].head()
ted_data['languages'] = ted_data['languages'].replace(0, 1)
ted_data['Talk_ID'] = range(1, len(ted_data)+1)
rating_names = set()
for index, row in ted_data.iterrows():
    rating = ast.literal_eval(row['ratings'])
    for item in rating:
        rating_names.add(item['name'])
    
print (rating_names)
rating_data = defaultdict(list)
for index, row in ted_data.iterrows():
    rating = ast.literal_eval(row['ratings'])
    rating_data['Talk_ID'].append(row['Talk_ID'])
    names = set()
    for item in rating:
        rating_data[item['name']].append(item['count'])
        names.add(item['name'])

rating_data = pd.DataFrame(rating_data)

rating_data.head()
rating_data['total'] = rating_data.sum(axis = 1)
rating_data = rating_data.sort_values('total', ascending=False)  
def column_percentage(column):
    return (column/rating_data['total'])*100

rating_data.loc[:, (rating_data.columns != 'total') &  (rating_data.columns !='Talk_ID')] = \
    rating_data.loc[:, (rating_data.columns != 'total') &  (rating_data.columns !='Talk_ID')].apply(column_percentage)

print (rating_data.head())
tags_data = defaultdict(list)
for index, row in ted_data.iterrows():
    themes = ast.literal_eval(row['tags'])
    for item in themes:
        tags_data['Talk_ID'].append(row['Talk_ID'])
        tags_data['tags'].append(item)

tags_data = pd.DataFrame(tags_data)

print (len(tags_data))
print (len(tags_data['tags'].unique()))
tags_data.head()
problemchars = re.compile(r'[=\+/&<>;\"\-\?%#$@\,\t\r\n]| and ')

problems_occupation = defaultdict(list)
for index, row in ted_data.iterrows():
    occupation = row['speaker_occupation']
    char = problemchars.search(occupation)
    if char:
        chars = char.group()
        problems_occupation[chars].append(occupation)
        
problems_occupation.keys()
problems_occupation['/']
mult_occupation = re.compile(r'\/|\,|\;|\+| and ')
end_issue = re.compile(r' \.\.\.')
occupation_data = defaultdict(list)
ignore_cases_list = ['HIV/AIDS fighter','9/11 mothers']

for index, row in ted_data.iterrows():
    occupation = row['speaker_occupation']
    problem_found = False
    if mult_occupation.search(occupation):
        problem_found = True
    if problem_found & (occupation not in ignore_cases_list):
        occupation = re.split('\/|\,|\;|\+| and ', occupation)
        for item in occupation:
            occupation_data['Talk_ID'].append(row['Talk_ID'])
            if end_issue.search(item):
                item = item.strip(' ...')
            occupation_data['speaker_occupation'].append(item.strip().lower())
    #All strings were converted to lowercase in order to avoid the same word in different formats.
    else:
        occupation_data['Talk_ID'].append(row['Talk_ID'])
        occupation_data['speaker_occupation'].append(occupation.lower())

occupation_data = pd.DataFrame(occupation_data)

print (occupation_data['speaker_occupation'].value_counts().head())
print ((len(occupation_data)))
print (len(occupation_data['speaker_occupation'].unique()))
occupation_data.head()
ted_basic_info = ted_data[['Talk_ID', 'title', 'duration','comments','views','languages']]
count_talks = defaultdict(list)
for rating in rating_data.columns:
    if (rating != 'Talk_ID') & (rating != 'total'):
        count_talks['rating'].append(rating) 
        count_talks['count'].append(rating_data[rating_data[rating] >0][rating].count())
    

sns.barplot(x="rating", y="count", data=count_talks)
plt.ylim(2400, 2600)
plt.xticks(rotation='vertical')
rating_values = rating_data.loc[:, (rating_data.columns != 'total') &  (rating_data.columns !='Talk_ID')] 
rating_data['main_rating'] = rating_values.apply(np.argmax, axis = 1)

rating_data['main_rating'].head()
label_order = rating_data['main_rating'].value_counts().sort_values(ascending = False).index

f, ax = plt.subplots(figsize=(12, 8))
ax = sns.countplot(x = 'main_rating', data = rating_data, order = label_order)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:d}'.format(height),
            ha="center") 
plt.xticks(rotation=30)
rating_values_updated = rating_data.loc[:, (rating_data.columns != 'total') &  (rating_data.columns !='Talk_ID')]
plt.figure(figsize=(20,15))
parallel_coordinates(rating_values_updated, 'main_rating', colormap=plt.get_cmap("tab20"))
plt.xlabel("Rating")
plt.ylabel("Percentage of Votes")
print (rating_data['total'].head(1))
print (rating_data['total'].tail(1))
rating_and_basic = ted_basic_info.merge(rating_data, how = 'left', on = ['Talk_ID'])
sns.regplot(x="views", y="total", data=rating_and_basic)
plt.xlabel("views")
plt.ylabel("Total Votes")
sns.regplot(x="comments", y="total", data=rating_and_basic)
plt.xlabel("comments")
plt.ylabel("Total Votes")
ted_basic_info.sort_values('views', ascending = False).head()
ted_basic_info.sort_values('comments', ascending = False).head()
list_ordered_by_median = rating_and_basic.groupby('main_rating')['views'].median().sort_values(ascending = False).index

f, ax = plt.subplots(figsize=(12, 8))
ax = sns.boxplot(x="main_rating", y="views", data=rating_and_basic, order = list_ordered_by_median)
plt.xticks(rotation=30)
plt.ylim((0,10000000))
list_ordered_by_median_com = rating_and_basic.groupby('main_rating')['comments'].median().sort_values(ascending = False).index

f, ax = plt.subplots(figsize=(12, 8))
ax = sns.boxplot(x="main_rating", y="comments", data=rating_and_basic, order = list_ordered_by_median_com)
plt.xticks(rotation=30)
plt.ylim((0,1000))
list_ordered_by_median_lan = rating_and_basic.groupby('main_rating')['languages'].median().sort_values(ascending = False).index

f, ax = plt.subplots(figsize=(12, 8))
ax = sns.boxplot(x="main_rating", y="languages", data=rating_and_basic, order = list_ordered_by_median_lan)
plt.xticks(rotation=30)
list_ordered_by_median_dur = rating_and_basic.groupby('main_rating')['duration'].median().sort_values(ascending = False).index

f, ax = plt.subplots(figsize=(12, 8))
ax = sns.boxplot(x="main_rating", y="duration", data=rating_and_basic, order = list_ordered_by_median_dur)
plt.xticks(rotation=30)
plt.ylim((0,40))
#Features to remove from the plot:
features_to_remove = ['Talk_ID', 'title']
corr = rating_and_basic.drop(features_to_remove, axis = 1).corr() 
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.title('Heatmap of Correlation between Features')
plt.scatter(rating_and_basic['Persuasive'], rating_and_basic['comments'], alpha = 0.5)
plt.ylabel("comments")
plt.xlabel("Persuasive percentage")
pearsonr(rating_and_basic['Persuasive'], rating_and_basic['comments'])
tags_rating = rating_data.merge(tags_data, how = 'left', on = ['Talk_ID'])
occupation_rating = rating_data.merge(occupation_data, how = 'left', on = ['Talk_ID'])
tags_rating.head()
occupation_rating.head()
def find_common_var(rating, var):
    #finds the main tags or speakers occupation
    if var == 'tags':
        count_tags = tags_rating[tags_rating['main_rating'] == rating]['tags'].value_counts()
    else:
        count_tags = occupation_rating[occupation_rating['main_rating'] ==\
                                       rating]['speaker_occupation'].value_counts()
    return count_tags
def main_tags_or_occupation(search):
    #var should be tags or speakers_occupation
    #Find the five main tags/occupation for each main rating and save in a set
    main_var_set = set()
    for name in rating_names:
        main_var_set.update(find_common_var(name, search).head().index)

    #Create a dataframe with the tags/occupation and values for each main rating
    rating_main_var = defaultdict(list)
    for name in rating_names:
        current_rating_table = find_common_var(name, search).head()
        main_var = current_rating_table.index
        rating_main_var['rating'].append(name)
        for var in main_var_set:
            if var not in main_var:
                rating_main_var[var].append(0)
            else:
                rating_main_var[var].append(current_rating_table[var])
    return rating_main_var

rating_main_tags = main_tags_or_occupation('tags')
rating_main_tags = pd.DataFrame(rating_main_tags)  
#defining the rating column as the new dataframe index:
rating_main_tags = rating_main_tags.set_index('rating')
rating_main_tags.head()
f, ax = plt.subplots(figsize=(15, 12))
cmap = sns.cubehelix_palette(light=1, as_cmap=True)
ax = sns.heatmap(rating_main_tags, annot = True, fmt = "d", cmap = cmap)
tags_data['tags'].value_counts().head(10)
rating_main_occup = main_tags_or_occupation('speakers_occupation')
rating_main_occup = pd.DataFrame(rating_main_occup)  

#defining the rating column as the new dataframe index:
rating_main_occup = rating_main_occup.set_index('rating')
f, ax = plt.subplots(figsize=(15, 12))
cmap = sns.cubehelix_palette(light=1, as_cmap=True)
ax = sns.heatmap(rating_main_occup, annot = True, fmt = "d", cmap = cmap)