#SETUP

import matplotlib.pyplot as plt

import pandas as pd 

import numpy as np



ted_df = pd.read_csv("../input/ted_main.csv")

ted_trans_df = pd.read_csv('../input/transcripts.csv')

combine = [ted_df, ted_trans_df]



#ted_df is the only data set right now



print(ted_df.columns.values)

df=pd.DataFrame(ted_df)

df



#Statistical Measuremeants

avg_views=df['views'].mean()



avg_views

median_views= df['views'].median()



median_views
avg_comments=df['comments'].mean()



avg_comments
median_comments=df['comments'].median()

median_comments
speaker_views_agg = df.groupby(['main_speaker']).agg({'views':sum})

top_20_speakers = speaker_views_agg.sort_values('views', ascending=False).head(20)

top_20_speakers
talk_views=df[['name','views']]

sorted_views=talk_views.sort_values('views', ascending=False)

top_20_tlk=sorted_views.head(20)

(top_20_tlk)



talk_views=df[['name','views']]

bottom_views=talk_views.sort_values('views', ascending=True)

btm_20_tlk=bottom_views.head(20)

(btm_20_tlk)
cmnt_views=df[['comments','views','name']]

sorted_cmnt=cmnt_views.sort_values('comments', ascending=False)

top_20_cmnt=sorted_cmnt.head(20)

(top_20_cmnt)

df_views=pd.DataFrame(sorted_views)

df_views









sum_of_views = sum(df_views['views']) #4330658578



sum_of_talks = df_views['name'].count() #2550



df_views['views_of_total'] = df_views['views']/sum_of_views



df_views['cumulative_views'] = np.cumsum(df_views.views_of_total)



df_views['counter'] = range(len(df_views['name']))



df_views['perc_talks_of_total'] = df_views['counter']/sum_of_talks



df_views




top_20_p= df_views['perc_talks_of_total']<= (0.2)



p_views= df_views['cumulative_views']<= (0.8)



pareto_views=df_views[top_20_p & p_views]

pareto_views.tail()



df_comments=pd.DataFrame(sorted_cmnt)



df_comments
sum_of_comments = sum(df_comments['comments']) #488484



sum_of_talks = df_comments['name'].count() #2550

df_comments['comments_of_total'] = df_comments['comments']/sum_of_comments



df_comments['cumulative_comments'] = np.cumsum(df_comments.comments_of_total)



df_comments['counter'] = range(len(df_comments['name']))



df_comments['perc_talks_of_total'] = df_comments['counter']/sum_of_talks





df_comments
top_20_p_c= df_comments['perc_talks_of_total']<= (0.2)



p_comments= df_comments['cumulative_comments']<= (0.8)



pareto_comments=df_comments[top_20_p_c & p_comments]

pareto_comments.tail()
cmnt_view_corr=(df_comments['comments'].corr(df_views['views']))

cmnt_view_corr