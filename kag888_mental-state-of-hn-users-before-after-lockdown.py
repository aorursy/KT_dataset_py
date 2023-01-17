# Lockdown start, no year b/c we need the date to get the period of the prior year
LOCKDOWN_START_DATE = '03-31'

# Posts to be dropped under this comment threshold
COMMENT_THRESHOLD = 10

# Only posts that match this regex to be included 
REG = 'depression|suicide|anxiety|burn?out|ptsd|ketamine|loneliness|isolation|sleep?deprivation'
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

hn = pd.read_csv('/kaggle/input/all-hacker-news-posts-stories-askshow-hn-polls/hn.csv')
print(f"Latest entry in the HN dataset is from {hn['Created At'].max()}")

hn_temp = hn[((hn['Created At'] > f"2019-{LOCKDOWN_START_DATE}") & (hn['Created At'] < '2019-09-17')) | (hn['Created At'] > f"2020-{LOCKDOWN_START_DATE}")]

hn2 = hn_temp.assign(in_lockdown=hn['Created At'] > '2020-03-31').dropna(subset=['Title'])

hn2.groupby('in_lockdown').describe()
hn3 = hn2[hn2.Title.str.contains(REG) ]

hn5 = hn3[hn3['Number of Comments'] > COMMENT_THRESHOLD]

hn5.groupby('in_lockdown').describe()
expl = 'with depression related keywords in the title after lockdown vs before'

post_diff = (hn5[hn5['in_lockdown'] == True].Title.count() / hn5[hn5['in_lockdown'] == False].Title.count() - 1) * 100
print(f"{round(post_diff)}% more submissions {expl}\n")

comment_diff = (hn5[hn5['in_lockdown'] == True]['Number of Comments'].mean() / hn5[hn5['in_lockdown'] == False]['Number of Comments'].mean() -1) * 100
print(f"{round(comment_diff)}% more comments on posts {expl}\n")

point_diff = (hn5[hn5['in_lockdown'] == True]['Points'].mean() / hn5[hn5['in_lockdown'] == False]['Points'].mean() -1) * 100
print(f"{round(point_diff)}% more upvotes on posts {expl}")