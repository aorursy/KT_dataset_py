import pandas as pd
import numpy as np
import seaborn as sns
sns.set_style('darkgrid')
times_df = pd.read_csv('/kaggle/input/world-university-rankings/timesData.csv' , thousands = ",")
shanghai_df = pd.read_csv('/kaggle/input/world-university-rankings/shanghaiData.csv')
times_df.head()
times_df.describe()
shanghai_df.head()
shanghai_df.describe()
def extract_info(input_df,name):
    df  = input_df.copy()
    info_df = pd.DataFrame({'nb_rows':df.shape[0], 'nb_columns': df.shape[1], 'name': name}, index = range(1))
    return info_df
all_info = pd.concat([times_df.pipe(extract_info,'times'), shanghai_df.pipe(extract_info,'shanghai')])
all_info
def clean_world_rank(input_df):
    df = input_df.copy()
    df.world_rank = df.world_rank.str.split('-').str[0].str.split('=').str[0]
    return df
common_col = set(shanghai_df.columns) & set(times_df.columns)
list(common_col)
def filter_year(input_df,years):
    df = input_df.copy()
    return df.query('year in {}'.format(list(years)))
common_years = set(times_df.year) & set(shanghai_df.year)
clean_times_df = times_df.loc[:,common_col].pipe(filter_year,common_years).pipe(clean_world_rank).assign(name='times')
clean_shanghai_df = shanghai_df.loc[:,common_col].pipe(filter_year,common_years).pipe(clean_world_rank).assign(name='shanghai')
ranking_df = pd.concat([clean_times_df,clean_shanghai_df])
ranking_df
pd.isnull(ranking_df.total_score).sum()/len(ranking_df)
ranking_df.drop('total_score', axis = 1, inplace = True)
ranking_df
ranking_df.info()
ranking_df.info(memory_usage = 'deep')
# Cast `world_rank` as type `int16`
ranking_df.world_rank = ranking_df.world_rank.astype('int16')

# Cast `unversity_name` as type `category`
ranking_df.university_name = ranking_df.university_name.astype('category')

# Cast `name` as type `category`
ranking_df.name = ranking_df.name.astype('category')
ranking_df.info(memory_usage='deep')
print(ranking_df.query("university_name == 'Massachusetts Institute of Technology (MIT)'"))
ranking_df.loc[lambda df: df.university_name == 'Massachusetts Institute of Technology (MIT)', 'university_name'] = 'Massachusetts Institute of Technology'
print(ranking_df.query("university_name == 'Massachusetts Institute of Technology'"))
top5_df = ranking_df.loc[lambda df : df.world_rank.isin(range(1,6)) , :]
top5_df.head()
def compute_set_similarity(df):
    pivoted = df.pivot(values = 'world_rank', columns = 'name', index = 'university_name').dropna()
    set_similarity = 100 * len(set(pivoted['shanghai'].index) & set(pivoted['times'].index))/5
    return set_similarity
grouped_df = top5_df.groupby('year')
set_similarity_df = pd.DataFrame({'set_similarity' : grouped_df.apply(compute_set_similarity)}).reset_index()
set_similarity_df
import matplotlib.pyplot as plt
%matplotlib inline
shanghai_df.plot.scatter('total_score', 'alumni', c='year', colormap='viridis')
plt.show()
times_df.country = times_df.country.replace('United States of America', 'USA').replace('United Kingdom', 'UK')
count_df = times_df['country'].value_counts()[:10]
count_df
count_df = count_df.reset_index()
count_df
count_df.columns = ['country', 'count']
count_df
sns.barplot(x = 'country', y = 'count', data = count_df)
sns.despine()

plt.show()