import pickle, os
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import seaborn as sns
from sklearn.linear_model import LinearRegression

def get_day(s):
    return str(datetime.fromtimestamp(s))[:-9]

def round_wk(i):
    return int(i / (60*60*24*7)) * 60*60*24*7
r_obs = pd.read_csv('/kaggle/input/most-viewed-memes-templates-of-2018/reddit_observations.csv',index_col='reddit_obs_num')
i_obs = pd.read_csv('/kaggle/input/most-viewed-memes-templates-of-2018/imgur_observations.csv',index_col='imgur_obs_num')
posts = pd.read_csv('/kaggle/input/most-viewed-memes-templates-of-2018/reddit_posts.csv',index_col='meme_id')
link_subreddit = posts[['reddit_post_id','subreddit']].drop_duplicates().set_index('reddit_post_id')['subreddit']

i_obs = i_obs.join(r_obs['upvotes'], on='reddit_obs_num')
i_obs = i_obs.join(link_subreddit, on='reddit_post_id').sort_values('timestamp')
first_obs = i_obs.drop_duplicates(subset='reddit_post_id', 
                                  keep='first').set_index('reddit_post_id')[['upvotes','imgur_viewcount']].sort_index()
last_obs = i_obs.drop_duplicates(subset='reddit_post_id', 
                                 keep='last').set_index('reddit_post_id')[['upvotes','imgur_viewcount']].sort_index()
delta = last_obs - first_obs
delta = delta[(delta['upvotes']>0)&(delta['imgur_viewcount']>0)]
delta = delta.join(link_subreddit)
delta['ratio'] = delta['imgur_viewcount']/delta['upvotes']
subreddit_list = posts['subreddit'].unique()
subreddit_ratios = pd.DataFrame(columns=['ratio','n'])

lr = LinearRegression(fit_intercept=True, normalize=False) 

for s in subreddit_list: 
    y = delta.loc[delta['subreddit']==s, 'imgur_viewcount']
    if len(y)>0:
        X = delta.loc[delta['subreddit']==s, 'upvotes'].values.reshape(-1,1)
        lr.fit(X, y)
        subreddit_ratios.loc[s,'ratio'] = lr.coef_[0]
    subreddit_ratios.loc[s,'n'] = len(y)

subreddit_ratios['average'] = subreddit_ratios.loc[subreddit_ratios['n']>100,'ratio'].mean()

def corrected_coef(c,n,av,lo_thr=10,hi_thr=100):
    if n < lo_thr: return av
    if n > hi_thr: return c
    w1 = hi_thr - n
    w2 = n - lo_thr
    return np.average([av,c], weights=[w1,w2])
        
subreddit_ratios['adjusted_ratio'] = [corrected_coef(c,n,av) for c,n,av in subreddit_ratios[['ratio','n','average']].values]
ratios = subreddit_ratios['adjusted_ratio'].to_dict()
ax = subreddit_ratios['adjusted_ratio'].sort_values().plot(kind='bar', figsize=(12,5))
t2 = ax.set_title('Estimated views per upvote, for each Subreddit')
link_post = posts[['reddit_post_id','subreddit','meme_template']].drop_duplicates().set_index('reddit_post_id')
r_obs = r_obs.join(link_post, on='reddit_post_id').sort_values('timestamp')
r_obs['basic_estimated_views'] = [u*ratios[s] for u,s in r_obs[['upvotes','subreddit']].values]
max_obs = r_obs.sort_values('basic_estimated_views').drop_duplicates(subset=['reddit_post_id','meme_template','subreddit'], keep='last')
template_x_subreddit = max_obs.pivot_table(columns='meme_template', index='subreddit', values='basic_estimated_views', aggfunc='sum')
template_concerntration = template_x_subreddit.max()/template_x_subreddit.sum()
template_propagation = (1 / template_concerntration.apply(lambda x: max([x,0.4]))).to_dict()
r_obs['complex_estimated_views'] = [v*template_propagation[t] for v,t in r_obs[['basic_estimated_views','meme_template']].values]
r_obs['day'] = r_obs['timestamp'].apply(get_day)
day_x_post = r_obs.pivot_table(index='day',columns='reddit_post_id',aggfunc='max',values='complex_estimated_views')
day_x_post = day_x_post.interpolate(limit_direction='forward').replace(np.nan,0)
day_x_post_delta = day_x_post.diff()
day_x_post_delta[day_x_post_delta < 0] = 0
day_x_post_delta.iloc[0] = day_x_post.iloc[0]
day_x_post_delta = day_x_post_delta.T.join(posts.set_index('reddit_post_id')['meme_template'])
day_x_template_delta = day_x_post_delta.groupby('meme_template').sum()
daily_views = day_x_template_delta.astype(int)
daily_views.loc[['harold','stefan_pref']]
cumulative_views = daily_views.T.cumsum()
ax = cumulative_views[['harold','stefan_trickery']].plot(
            figsize=(12,5), ylim=(0), xlim=(0,364)) 
t = ax.set_ylabel('Cumulative views', fontsize=12)
# RIP Stefan
total_views = cumulative_views.iloc[-1].sort_values(ascending=False)
fig,ax = plt.subplots(figsize=(12,5))
ax.bar(x=range(len(total_views)), height=total_views)
ax.set_yticklabels([str(int(i/1000000))+'m' for i in ax.get_yticks()])
ax.set_xlim(-1,250)
ax.set_xticks([0,49,99,149,199,249])
ax.set_xticklabels([1,50,100,150,200,250])
ax.set_ylabel('Views throughout 2018', fontsize=14)
ax.set_xlabel('Meme templates (ranked by views)', fontsize=14)
t = ax.text(130, 1.3e8, color='r', fontsize=12,
            s='FUN FACT:\nThe top meme template has more views\nthan the templates ranked 200-250 combined')
plt.tight_layout()
total_views.iloc[0] > total_views.iloc[-51:].sum() 