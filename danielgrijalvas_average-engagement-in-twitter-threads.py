import pandas as pd

csv = pd.read_csv('../input/fifteen_twenty.csv', encoding='iso-8859-1')
csv.head()
grouped_by_thread = csv.groupby(['thread_number'])

retweets = {}
likes = {}
replies = {}

# retweets
for thread, data in dict(list(grouped_by_thread)).items():
    retweets[thread] = list(data['retweets'])

retweets_by_thread = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in retweets.items()]))

# likes
for thread, data in dict(list(grouped_by_thread)).items():
    likes[thread] = list(data['likes'])

likes_by_thread = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in likes.items()]))

# replies
for thread, data in dict(list(grouped_by_thread)).items():
    replies[thread] = list(data['replies'])

replies_by_thread = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in replies.items()]))

retweets_by_thread.head()
average_length = grouped_by_thread.size().mean()
average_length
from bokeh.io import output_notebook, show
from bokeh.plotting import figure
from bokeh.models import Span

output_notebook()
# averages
avg = pd.DataFrame()
avg['Retweets'] = retweets_by_thread.mean(axis=1)
avg['Likes'] = likes_by_thread.mean(axis=1)
avg['Replies'] = replies_by_thread.mean(axis=1)

average_engagement = figure(plot_width=700, 
           plot_height=350, 
           title='Average engagement in 15-20 tweet-long Twitter threads', 
           background_fill_color="#f2f3f7", 
           y_axis_label='Engagement (# of interactions)', 
           x_axis_label='Tweets along the thread')

average_engagement.line(list(range(1,21)), avg['Retweets'].values,line_color='#17bf63', legend='Retweets')
average_engagement.line(list(range(1,21)), avg['Likes'].values, line_color='#e0245e', legend='Likes')
average_engagement.line(list(range(1,21)), avg['Replies'].values, line_color='#1da1f2', legend='Replies')

show(average_engagement)
from bokeh.layouts import row as bokeh_row

scatter_rts = figure(plot_width=420, plot_height=310, title='Scatter plot of retweets in 15-20 tweet-long threads', x_axis_label='Tweets along the thread', y_axis_label='# of Retweets')
scatter_likes = figure(plot_width=420, plot_height=310, title='Scatter plot of likes in in 15-20 tweet-long threads', x_axis_label='Tweets along the thread', y_axis_label='# of Likes')
scatter_replies = figure(plot_width=420, plot_height=310, title='Scatter plot of replies in 15-20 tweet-long threads', x_axis_label='Tweets along the thread', y_axis_label='# of Replies')

# add each data point to the retweets scatter plot
for row in retweets_by_thread:
    scatter_rts.circle(list(range(1,21)), retweets_by_thread.loc[:, row], size=3, line_color="#17bf63", fill_color="#17bf63", fill_alpha=0.5)

# add each data point to the likes scatter plot    
for row in likes_by_thread:
    scatter_likes.circle(list(range(1,21)), likes_by_thread.loc[:, row], size=3, line_color="#e0245e", fill_color="#e0245e", fill_alpha=0.5)

# add each data point to the replies scatter plot
for row in replies_by_thread:
    scatter_replies.circle(list(range(1,21)), replies_by_thread.loc[:, row], size=3, line_color="#1da1f2", fill_color="#1da1f2", fill_alpha=0.5)
    
show(bokeh_row(scatter_rts, scatter_likes, scatter_replies))
import numpy as np

# retweets
hist_rts_values, rt_edges = np.histogram(retweets_by_thread.iloc[0, :])
hist_rts = figure(plot_width=420, plot_height=310, 
                  title='Histogram of retweets in the first tweet of each thread', 
                  x_axis_label='# of Retweets', 
                  y_axis_label='Frequency')

hist_rts.quad(top=hist_rts_values, bottom=0, left=rt_edges[:-1], right=rt_edges[1:],
        fill_color="#17bf63", line_color="#17bf63")

# likes
hist_likes_values, likes_edges = np.histogram(likes_by_thread.iloc[0, :])
hist_likes = figure(plot_width=420, plot_height=310, 
                  title='Histogram of likes in the first tweet of each thread', 
                  x_axis_label='# of Likes', 
                  y_axis_label='Frequency')

hist_likes.quad(top=hist_likes_values, bottom=0, left=likes_edges[:-1], right=likes_edges[1:],
        fill_color="#e0245e", line_color="#e0245e")

# replies
hist_rpl_values, rpl_dges = np.histogram(replies_by_thread.iloc[0, :])
hist_replies = figure(plot_width=420, plot_height=310, 
                  title='Histogram of replies in the first tweet of each thread', 
                  x_axis_label='# of Replies', 
                  y_axis_label='Frequency')

hist_replies.quad(top=hist_rpl_values, bottom=0, left=rpl_dges[:-1], right=rpl_dges[1:],
        fill_color="#1da1f2", line_color="#1da1f2")

# show results
show(bokeh_row(hist_rts, hist_likes, hist_replies))
# median of engagement
median = pd.DataFrame()
median['Retweets'] = retweets_by_thread.median(axis=1)
median['Likes'] = likes_by_thread.median(axis=1)
median['Replies'] = replies_by_thread.median(axis=1)

median_engagement = figure(plot_width=700, 
           plot_height=350, 
           title='Median of engagement in 15-20 tweet-long Twitter threads', 
           background_fill_color="#f2f3f7", 
           y_axis_label='Engagement (# of interactions)', 
           x_axis_label='Tweets along the thread')

# add a line renderer
median_engagement.line(list(range(1,21)), median['Retweets'].values,line_color='#17bf63', legend='Retweets')
median_engagement.line(list(range(1,21)), median['Likes'].values, line_color='#e0245e', legend='Likes')
median_engagement.line(list(range(1,21)), median['Replies'].values, line_color='#1da1f2', legend='Replies')

show(bokeh_row(median_engagement, average_engagement))