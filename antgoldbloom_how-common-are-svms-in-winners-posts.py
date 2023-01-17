import re

import pandas as pd 

from IPython.display import display

from IPython.display import HTML

pd.set_option('max_colwidth',1000)
##Load data

df_winner_post = pd.read_csv('../input/WinnersInterviewBlogPosts.csv') 

## convert publication date to datetime type

df_winner_post.index = pd.to_datetime(df_winner_post['publication_date'])

del(df_winner_post['publication_date'])
df_winner_post['SVM'] = df_winner_post['content'].str.count('SVM',flags=re.IGNORECASE) + df_winner_post['content'].str.count('support vector machine',flags=re.IGNORECASE) > 0

df_winner_post_with_SVM = df_winner_post[['title','link']][df_winner_post['SVM'] == True]

display(df_winner_post_with_SVM)