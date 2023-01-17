# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from sklearn.cluster import DBSCAN

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.cluster import KMeans

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# for visualizations
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
%time df = pd.read_csv('/kaggle/input/sites-information-data-from-alexacom-dataset/alexa.com_site_info.csv')

print(df.shape)
df.head()
df.dtypes
X = df[['keyword_opportunities_breakdown_optimization_opportunities', 'keyword_opportunities_breakdown_keyword_gaps']].dropna().to_numpy()
clustering = DBSCAN(eps=3, min_samples=2).fit(X)
# predictions = np.array(kmeans.predict(X))
# prediction_array = [[], []]
# for i in range(2):
#     if predictions[i] == i:
#         prediction_array[i].append(X[i])
# prediction_array_1 = np.array(prediction_array[0])

# plt.plot(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,-1], 'co')
# plt.show()
# print(kmeans.cluster_centers_)
df.dtypes
X = df[['keyword_opportunities_breakdown_optimization_opportunities', 'keyword_opportunities_breakdown_keyword_gaps']].dropna().to_numpy()
kmeans = KMeans(n_clusters=3).fit(X)
predictions = np.array(kmeans.predict(X))
prediction_array = [[], []]
for i in range(2):
    if predictions[i] == i:
        prediction_array[i].append(X[i])
prediction_array_1 = np.array(prediction_array[0])
plt.plot(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,-1], 'co')
plt.show()
print(kmeans.cluster_centers_)
plt.scatter(x=x, y=y)
plt.show()
def category(x):
    return df[df['category'] == x][['site link','keyword_opportunities_breakdown_optimization_opportunities','keyword_opportunities_breakdown_keyword_gaps','keyword_opportunities_breakdown_easy_to_rank_keywords', 'keyword_opportunities_breakdown_buyer_keywords', 'This_site_rank_in_global_internet_engagement', 'Daily_time_on_site']]


# let's check the Indian Players 
x = category('Adult/Arts')
x.shape
df.isnull().sum()
################ Not important !!!
last_rank = df['This_site_rank_in_global_internet_engagement']
def search_engine_optimization(data):
    return int(round((data[['Daily_time_on_site', 'all_topics_easy_to_rank_keywords_relevance_to_site_parameter_1', 
                               'audience_overlap_sites_overlap_scores_parameter_1']].mean()).mean()))

# df['This_site_rank_in_global_internet_engagement'] = df.apply(search_engine_optimization, axis = 1)
# print(last_rank.corr(df.apply(search_engine_optimization, axis = 1)))
df['This_site_rank_in_global_internet_engagement'] = df.apply(search_engine_optimization, axis = 1)
print(last_rank)
print('--------------------------------------')
print(df['This_site_rank_in_global_internet_engagement'])
# comparison of preferred foot over the different players
plt.xticks(rotation=90)
plt.rcParams['figure.figsize'] = (200, 20)
sns.countplot(df['category'], palette = 'pink')
plt.title('Most Preferred Foot of the Players', fontsize = 7)
plt.show()
# different positions acquired by the players 
plt.figure(figsize = (18, 8))
plt.xticks(rotation=90, fontsize=3)
plt.style.use('fivethirtyeight')
ax = sns.countplot('category', data = df, palette = 'bone')
ax.set_xlabel(xlabel = 'Different Positions in Football', fontsize = 16)
ax.set_ylabel(ylabel = 'Count of Players', fontsize = 16)
ax.set_title(label = 'Comparison of Positions and Players', fontsize = 20)
plt.show()
# Comparing categories

import warnings
warnings.filterwarnings('ignore')

numeric_columns = []
str_counter = 0
for item in df:
    try:
        y = 0
        x = df[item][1] * 2
        z = x + y
        numeric_columns.append(item)
    except:
        str_counter += 1
print(str_counter)

for i in range(len(numeric_columns)):
    plt.rcParams['figure.figsize'] = (15, 5)
    sns.distplot(df[numeric_columns[i]], color = 'blue')
    plt.xlabel(numeric_columns[i], fontsize = 16)
    plt.ylabel('Count', fontsize = 16)
    plt.title('Distribution of ' + str(numeric_columns[i]), fontsize = 20)
    plt.xticks(rotation = 90)
    plt.show()


# To show Different catgories

plt.style.use('dark_background')
df['category'].value_counts().head(80).plot.bar(color = 'orange', figsize = (20, 7))
plt.title('Different categories web sites', fontsize = 30, fontweight = 20)
plt.xlabel('Name of The Category')
plt.ylabel('count')
plt.show()
numeric_columns = []
str_counter = 0
for item in df:
    try:
        y = 0
        x = df[item][1] * 2
        z = x + y
        numeric_columns.append(item)
    except:
        str_counter += 1
print(str_counter)

def plot_show():
    for i in range(len(numeric_columns)):
        for j in range(i + 1, len(numeric_columns) - 1):
            if j == 3:
                return 0
            plt.rcParams['figure.figsize'] = (20, 7)
            plt.style.use('seaborn-dark-palette')
            sns.boxenplot(df[numeric_columns[i]], df[numeric_columns[j]], palette = 'pink')
            plt.xticks(rotation = 90)
            plt.title('Comparison of ' + numeric_columns[i] + ' and ' + numeric_columns[j], fontsize = 10)
            plt.show()
            
plot_show()
numeric_columns = []
str_counter = 0
for item in df:
    try:
        y = 0
        x = df[item][1] * 2
        z = x + y
        numeric_columns.append(item)
    except:
        str_counter += 1
print(str_counter)

def plot_show():
    for i in range(len(numeric_columns)):
        for j in range(i + 1, len(numeric_columns) - 1):
            for k in range(j + 1, len(numeric_columns) - 1):
                if k == 6:
                    return 0
                plt.scatter(df[numeric_columns[i]], df[numeric_columns[j]], s = df[numeric_columns[k]]*0.05, c = 'pink')
                plt.xlabel('Overall Ratings', fontsize = 20)
                plt.ylabel('International Reputation', fontsize = 20)
                plt.title('Ratings vs Reputation', fontweight = 20, fontsize = 20)
                #plt.legend('Age', loc = 'upper left')
                plt.show()
                
plot_show()
df = df[[col for col in df if df[col].nunique() > 1 and ('audience_overlap_sites_overlap_scores_parameter_4' in col or '_keyword_gaps_Avg_traffic_parameter_3' in col or '_keyword_gaps_Avg_traffic_parameter_4' in col in col or 'comparison_metrics_data_' in col or '_relevance_to_site_parameter_1' in col or '_keywords_Avg_traffic_parameter_4' in col or 'rank_in_global_internet_engagement' in col or 'Daily_time_on_site' in col or 'keyword_opportunities_' in col or 'keyword_gaps_search_popularity_parameter_1' in col or '_easy_to_rank_keywords_search_pop_parameter_1' in col)]]
col_items = []
for item in df:
    col_items.append(item)

sns.heatmap(df[col_items].corr(), annot = True)

plt.title('Histogram of the Dataset', fontsize = 30)
plt.show()
some_countries = []
for item in df['category'].head(1000):
    if item not in some_countries:
        some_countries.append(item)

# some_countries = ['Adult/Arts', 'Adult/Shopping', 'Adult/Society', 'Arts/Animation', 'Arts/Design', 'Arts/Illustration', 'Business/Aerospace_and_Defense', 'Computers/Consultants']
data_countries = df.loc[df['category'].isin(some_countries) & df['This_site_rank_in_global_internet_engagement']]

plt.rcParams['figure.figsize'] = (15, 7)
ax = sns.violinplot(x = data_countries['category'], y = data_countries['This_site_rank_in_global_internet_engagement'], palette = 'Reds')
ax.set_xlabel(xlabel = 'Category', fontsize = 9)
ax.set_ylabel(ylabel = 'Rank', fontsize = 9)
ax.set_title(label = 'Distribution of catgory and rank', fontsize = 20)
plt.xticks(rotation = 90)
plt.show()
# Every Nations' Player and their overall scores

some_countries = []
for item in df['category'].head(1000):
    if item not in some_countries:
        some_countries.append(item)

# some_countries = ['Adult/Arts', 'Adult/Shopping', 'Adult/Society', 'Arts/Animation', 'Arts/Design', 'Arts/Illustration', 'Business/Aerospace_and_Defense', 'Computers/Consultants']
data_countries = df.loc[df['category'].isin(some_countries) & df['This_site_rank_in_global_internet_engagement']]

# # some_countries = ('England', 'Germany', 'Spain', 'Argentina', 'France', 'Brazil', 'Italy', 'Columbia')
# data_countries = data.loc[data['Nationality'].isin(some_countries) & data['Overall']]

plt.rcParams['figure.figsize'] = (15, 7)
ax = sns.barplot(x = data_countries['category'], y = data_countries['This_site_rank_in_global_internet_engagement'], palette = 'spring')
ax.set_xlabel(xlabel = 'Category', fontsize = 9)
ax.set_ylabel(ylabel = 'Overall Rank', fontsize = 9)
ax.set_title(label = 'Distribution of overall rank from different categories', fontsize = 20)
plt.xticks(rotation = 90)
plt.show()
# Distribution of Ages in some Popular clubs

some_clubs = []
for item in df['category'].head(1000):
    if item not in some_clubs:
        some_clubs.append(item)

# some_countries = ['Adult/Arts', 'Adult/Shopping', 'Adult/Society', 'Arts/Animation', 'Arts/Design', 'Arts/Illustration', 'Business/Aerospace_and_Defense', 'Computers/Consultants']
data_countries = df.loc[df['category'].isin(some_countries) & df['Daily_time_on_site']]

# some_clubs = ('CD Leganés', 'Southampton', 'RC Celta', 'Empoli', 'Fortuna Düsseldorf', 'Manchestar City',
#              'Tottenham Hotspur', 'FC Barcelona', 'Valencia CF', 'Chelsea', 'Real Madrid')

data_club = df.loc[df['category'].isin(some_clubs) & df['Daily_time_on_site']]

plt.rcParams['figure.figsize'] = (15, 8)
ax = sns.boxenplot(x = 'category', y = 'Daily_time_on_site', data = data_club, palette = 'magma')
ax.set_xlabel(xlabel = 'Category', fontsize = 10)
ax.set_ylabel(ylabel = 'Daily_time_on_site', fontsize = 10)
ax.set_title(label = 'Disstribution of Daily time on site in some categories', fontsize = 20)
plt.xticks(rotation = 90)
plt.show()
new_df = []
for i in range(len(df['This_site_rank_in_global_internet_engagement'])):
    df['This_site_rank_in_global_internet_engagement'] = 1 / df['This_site_rank_in_global_internet_engagement']
    df['Daily_time_on_site'] = 1 / df['Daily_time_on_site']

player_features = ('This_site_rank_in_global_internet_engagement','Daily_time_on_site', 'all_topics_easy_to_rank_keywords_relevance_to_site_parameter_1', 'audience_overlap_sites_overlap_scores_parameter_1')

# Top four features for every position in football

for i, val in df.groupby(df['category'].head(1000))[player_features].mean().iterrows():
    print('Position {}: {}, {}, {}'.format(i, *tuple(val.nlargest(4).index)))
from math import pi

idx = 1
plt.figure(figsize=(15,45))
for position_name, features in df.groupby(df['category'])[player_features].mean().iterrows():
    top_features = dict(features.nlargest(4))
    
    # number of variable
    categories=top_features.keys()
    N = len(categories)

    # We are going to plot the first line of the data frame.
    # But we need to repeat the first value to close the circular graph:
    values = list(top_features.values())
    values += values[:1]

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(5, 3, idx, polar=True)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories, color='grey', size=6, rotation= 90)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([25,50,75,100], ["25","50","75", "100"], color="grey", size=7, rotation= 90)
    plt.ylim(0,100)
    
    plt.subplots_adjust(hspace = 0.5)
    
    # Plot data
    ax.plot(angles, values, linewidth=1, linestyle='solid')

    # Fill area
    ax.fill(angles, values, 'b', alpha=0.1)
    
    plt.title(position_name, size=8, y=1.1)
    
    idx += 1 
sns.lmplot(x = 'all_topics_easy_to_rank_keywords_relevance_to_site_parameter_1', y = 'Daily_time_on_site', data = df.head(100), col = 'category')
plt.xticks(rotation = 90)
plt.show()
sns.lineplot(df['Daily_time_on_site'], df['This_site_rank_in_global_internet_engagement'], palette = 'Wistia')
plt.title('Daily_time_on_site vs Ranking', fontsize = 20)

plt.show()