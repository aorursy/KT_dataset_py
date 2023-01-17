import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
ted_df = pd.read_csv("../input/ted_main.csv")
print(ted_df.columns.values)

df = pd.DataFrame(ted_df)
print(df.describe())
views_mean = df['views'].mean()
views_mean
views_median = df['views'].median()
views_median
views_std = df['views'].std()
views_std
views_cv = views_std / views_mean
views_cv
plt.plot(df.views)
plt.title("Views per Talk")
comm_mean = df['comments'].mean()
comm_mean
comm_median = df['comments'].median() 
comm_median
comm_std = df['comments'].std()
comm_std
comm_cv = comm_std / comm_mean
comm_cv
plt.plot(df.views)
plt.title("Comments per Talk")
talk_views = df[['name', 'views']]
sorted_talks_by_views = talk_views.sort_values('views', ascending=False)
top_20_talks = sorted_talks_by_views.head(20)
top_20_talks
ax = plt.subplots(figsize=(15,10))
ax = sns.barplot(x='views', y='name', data=top_20_talks, palette='Reds_r')
plt.xlabel('views',fontsize = 15,color='red')
plt.ylabel('name',fontsize = 15,color='red')
plt.title("Top 20 Talks by Views")
cmnt_views = df[['views','name','comments']]
sorted_cmnt = cmnt_views.sort_values('comments', ascending=False)
top_20_cmnt_sort = sorted_cmnt.head(20)
top_20_cmnt_sort
ax = plt.subplots(figsize=(15,10))
ax = sns.barplot(x='comments', y='name', data=top_20_cmnt_sort, palette='Reds_r')
plt.xlabel('comments',fontsize = 15,color='red')
plt.ylabel('name',fontsize = 15,color='red')
plt.title("Top 20 Talks by Comments")
cmnt_view_corr = df['views'].corr(df['comments'])
cmnt_view_corr
fig, ax1 = plt.subplots()
x = [df['views']];
y = [df['comments']]
ax1.plot(x, y, 'bo')
plt.show()
# New df for cumulative views manipulation

df_views = pd.DataFrame(sorted_talks_by_views)
# aggregative variables

sum_of_views = sum(df_views['views']) #4330658578

sum_of_talks = df_views['name'].count() #2550
# New manipulated columns

df_views['views_of_total'] = df_views['views']/sum_of_views

df_views['cumulative_views'] = np.cumsum(df_views.views_of_total)

df_views['counter'] = range(len(df_views['name']))

df_views['perc_talks_of_total'] = df_views['counter']/sum_of_talks

df_views.head()
top_20_p = df_views['perc_talks_of_total'] <= 0.2 #20% of the talks

p_views = df_views['cumulative_views'] <= 0.8 #80% of the views

pareto_views = df_views[top_20_p & p_views]

pareto_views.tail(10)
p_views_80 = df_views['cumulative_views'] >= 0.8 #80% of the views

pareto_views_80 = df_views[p_views_80]

pareto_views_80.head(10)
# New df for cumulative views manipulation

df_comments = pd.DataFrame(sorted_cmnt)
# aggregative variables

sum_of_comments = sum(df_comments['comments']) #488484

sum_of_talks = df_comments['name'].count() #2550

# New manipulated columns

df_comments['comments_of_total'] = df_comments['comments']/sum_of_comments

df_comments['cumulative_comments'] = np.cumsum(df_comments.comments_of_total)

df_comments['counter'] = range(len(df_comments['name']))

df_comments['perc_talks_of_total'] = df_comments['counter']/sum_of_talks

df_comments.head()
top_20_p_c= df_comments['perc_talks_of_total']<= (0.2)

p_comments= df_comments['cumulative_comments']<= (0.8)

pareto_comments=df_comments[top_20_p_c & p_comments]

pareto_comments.tail(10)
p_comments_80 = df_comments['cumulative_comments'] >= 0.8 #80% of the views

pareto_comments_80 = df_comments[p_comments_80]

pareto_comments_80.head(10)