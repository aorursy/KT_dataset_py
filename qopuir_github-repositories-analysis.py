import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import squarify
from datetime import datetime
file_repos = "../input/github-repositories.csv"
df_repos = pd.read_csv(file_repos, index_col = False)
df_repos.head(5)
df_repos['id'].count()
df_repos_by_stars = df_repos.sort_values(by='stars', ascending=False).reset_index()
df_repos_by_stars[['full_name', 'stars']].head(10)
df_repos_by_lang = df_repos.groupby(['language'])['id'].agg(['count'])
df_repos_by_lang = df_repos_by_lang.sort_values(by = 'count', ascending = False).reset_index()
df_repos_by_lang.head(10)
df_repos_by_lang.sort_values(by = 'count', ascending = True).head(10)
total_repos = df_repos['id'].count()

df_repos_by_lang['percentage'] = df_repos_by_lang['count'].apply(lambda count: count / total_repos * 100)
df_repos_by_lang['new_language'] = df_repos_by_lang.apply(lambda row: 'Other' if row['percentage'] < 1 else row['language'], axis = 1)

df_repos_by_lang.head(5)
df_repos_by_lang_transformed = df_repos_by_lang.groupby(['new_language'])['count'].agg(['sum'])
df_repos_by_lang_transformed = df_repos_by_lang_transformed.sort_values(by='sum', ascending=False).reset_index()
df_repos_by_lang_transformed
labels = df_repos_by_lang_transformed['new_language']
values = df_repos_by_lang_transformed['sum']

fig, ax =  plt.subplots(figsize=(16, 9))

squarify.plot(sizes = values, label = labels, alpha=.8, ax = ax)

plt.axis('off')
plt.show()
df_repos['created_at_datetime'] = pd.to_datetime(df_repos['created_at'], format = "%Y-%m-%dT%H:%M:%SZ")
df_repos['created_at_year'] = df_repos['created_at_datetime'].apply(lambda created_at_datetime: created_at_datetime.strftime("%Y") if pd.notnull(created_at_datetime) else created_at_datetime)
df_repos_by_year = df_repos.groupby(['created_at_year'])['id'].agg(['count']).reset_index()
df_repos_by_year
x = df_repos_by_year['created_at_year']
y = df_repos_by_year['count']

fig, ax =  plt.subplots(figsize=(16, 9))

ax.bar(x, y, color = 'green')
ax.set_title('Repos created by year')
ax.set_xlabel('Year')
ax.set_ylabel('# of repos created')

plt.xticks(rotation = 90)
plt.plot()
df_repos['created_at_month'] = df_repos['created_at_datetime'].apply(lambda created_at_datetime: created_at_datetime.strftime("%Y-%m") if pd.notnull(created_at_datetime) else created_at_datetime)
df_repos_by_month = df_repos.groupby(['created_at_month'])['id'].agg(['count']).reset_index()
df_repos_by_month.head(12)
x = df_repos_by_month['created_at_month']
y = df_repos_by_month['count']

fig, ax =  plt.subplots(figsize=(16, 9))

ax.bar(x, y, color = 'green')
ax.set_title('Repos created by month')
ax.set_xlabel('Month')
ax.set_ylabel('# of repos created')

plt.xticks(rotation = 90)
plt.plot()
df_repos_forked = df_repos.groupby(['is_fork'])['id'].agg(['count']).reset_index()
df_repos_forked['label'] = df_repos_forked['is_fork'].apply(lambda is_fork: 'Fork' if is_fork else 'No Fork')
df_repos_forked
labels = df_repos_forked['label']
count = df_repos_forked['count']

fig, ax = plt.subplots(figsize = (8, 8))

ax.pie(count, labels = labels, autopct = '%1.1f%%')
ax.set_title('Fork repos')

plt.plot()
df_creators = df_repos.groupby(['owner_id', 'owner_name'])['owner_id'].agg(['count'])
df_creators_top_25 = df_creators.sort_values(by = 'count', ascending = False).head(25).reset_index()
df_creators_top_25.head(5)
df_repos_of_creators_top_25 = pd.merge(df_creators_top_25, df_repos, how = 'inner', on = ['owner_id'])
df_repos_by_creator = df_repos_of_creators_top_25.groupby(['owner_name_x', 'is_fork'])['owner_id'].agg(['count']).sort_values(by = ['owner_name_x', 'is_fork'], ascending = True).reset_index()
df_repos_by_creator.head(5)
df_repos_by_creator_transformed = df_repos_by_creator.pivot(index = 'owner_name_x', columns = 'is_fork', values = 'count').reset_index()
df_repos_by_creator_transformed[True].fillna(0, inplace = True)
df_repos_by_creator_transformed[False].fillna(0, inplace = True)
df_repos_by_creator_transformed.head(5)
N = df_creators_top_25['owner_id'].count()

fig, ax =  plt.subplots(figsize = (16, 9))

original_repos = np.array(df_repos_by_creator_transformed[False])
forked_repos = np.array(df_repos_by_creator_transformed[True])
total_repos = np.array(original_repos + forked_repos)

ind = np.arange(N) # the x locations for the groups
width = 0.25       # the width of the bars

s1 = ax.bar(ind - width, total_repos, width, color='green')
s2 = ax.bar(ind, original_repos, width, color = 'blue')
s3 = ax.bar(ind + width, forked_repos, width, color='red')

ax.set_title('Repos created by top 25 creators')
ax.set_xticks(ind + width / 3)
ax.set_xticklabels(np.array(df_repos_by_creator_transformed['owner_name_x']))
ax.legend((s1[0], s2[0], s3[0]), ('Total', 'No Fork', 'Fork'))

plt.xticks(rotation = 90)
plt.plot()
df_language_evolution = df_repos.groupby(['language', 'created_at_month'])['id'].agg(['count']).reset_index()
df_language_evolution.head(5)
df_language_evolution_filtered = df_language_evolution[df_language_evolution['language'].isin(df_repos_by_lang_transformed['new_language'])]
df_language_evolution_filtered = df_language_evolution_filtered.sort_values(by = ['language', 'created_at_month'], ascending = True)
df_language_evolution_filtered.head(5)
df_language_evolution_transformed = df_language_evolution_filtered.pivot(index = 'created_at_month', columns = 'language', values = 'count')
df_language_evolution_transformed.head(5)
df_language_evolution_transformed.fillna(0, inplace = True)
df_language_evolution_transformed.head(5)
df_language_evolution_transformed = df_language_evolution_transformed.cumsum().reset_index()
df_language_evolution_transformed.head(5)
N = df_language_evolution_transformed['created_at_month'].count()

fig, ax =  plt.subplots(figsize=(16, 9))

ind = np.arange(N)

legend_values = list()
legend_titles = list()

for column in df_language_evolution_transformed:
    if column != 'created_at_month':
        sn = ax.plot(np.array(df_language_evolution_transformed[column]))
        
        legend_values.append(sn[0])
        legend_titles.append(column)

ax.set_title('Evolución de los lenguajes')
ax.set_xlabel('Tiempo')
ax.set_xticks(ind)
ax.set_xticklabels(np.array(df_language_evolution_transformed['created_at_month']))
ax.set_ylabel('Repositorios creados')
ax.grid(True)
ax.legend(legend_values, legend_titles)

plt.xticks(rotation = 90)
plt.plot()