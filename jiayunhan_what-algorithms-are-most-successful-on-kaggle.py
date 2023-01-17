import re

import pandas as pd 

from IPython.display import display

import seaborn as sns

sns.set(style="white")
##Load data

df_winner_post = pd.read_csv('../input/WinnersInterviewBlogPosts.csv') 

## convert publication date to datetime type

df_winner_post.index = pd.to_datetime(df_winner_post['publication_date'])

del(df_winner_post['publication_date'])
#Look for which winners writeups refer to SVMs, Random Forests, Neural Nets and GBMs

df_winner_post['SVM'] = df_winner_post['content'].str.count('SVM',flags=re.IGNORECASE) + df_winner_post['content'].str.count('support vector machine',flags=re.IGNORECASE) > 0

df_winner_post['Random Forest'] = df_winner_post['content'].str.count('random forest',flags=re.IGNORECASE) + df_winner_post['content'].str.count('randomforest',flags=re.IGNORECASE) > 0

df_winner_post['Neural Net'] = df_winner_post['content'].str.count('deep learning',flags=re.IGNORECASE) + df_winner_post['content'].str.count('neural net',flags=re.IGNORECASE) + df_winner_post['content'].str.count('CNN',flags=re.IGNORECASE) > 0

df_winner_post['GBM'] = df_winner_post['content'].str.count('gbm',flags=re.IGNORECASE) + df_winner_post['content'].str.count('gradient boosting machine',flags=re.IGNORECASE) + df_winner_post['content'].str.count('xgboost',flags=re.IGNORECASE) > 0
winners_by_method = df_winner_post[['SVM','Random Forest','Neural Net','GBM']].sum()/len(df_winner_post)

ax = winners_by_method.plot(kind='bar')

ax.set_ylabel("% of posts")
#notice that we had few posts in 2013 and 2014

df_groupby = df_winner_post.groupby(df_winner_post.index.year)

df_groupby['title'].count()
#group 2013 and 2014 because we had so few posts for those two years

df_groupby_sum = df_groupby[['SVM','Random Forest','Neural Net','GBM']].sum()

df_groupby_sum[df_groupby_sum.index == 2013] = df_groupby_sum[df_groupby_sum.index == 2013].values + df_groupby_sum[df_groupby_sum.index == 2014].values

df_groupby_sum = df_groupby_sum.drop([2010,2014])

df_groupby_sum = df_groupby_sum.set_index([['2011','2012','2013 & 2014','2015','2016']])



df_groupby_count = df_groupby[['SVM','Random Forest','Neural Net','GBM']].count()

df_groupby_count[df_groupby_count.index == 2013] = df_groupby_count[df_groupby_count.index == 2013].values + df_groupby_count[df_groupby_count.index == 2014].values

df_groupby_count = df_groupby_count.drop([2010,2014])

df_groupby_count = df_groupby_count.set_index([['2011','2012','2013 & 2014','2015','2016']])
ax1 = (df_groupby_sum/df_groupby_count).plot()

ax1.set_ylabel = '% of posts'