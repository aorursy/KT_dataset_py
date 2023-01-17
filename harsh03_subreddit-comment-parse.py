# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install praw
import praw
import re
import os
from praw.models import MoreComments
def deEmojify(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)
# create a file to store the parsed comments
if os.path.isfile('output.csv'):
    f = open("output.csv", "a+", encoding = 'utf-8')
else:
    f = open("output.csv", "w+", encoding = 'utf-8') 

reddit = praw.Reddit(user_agent="Comment Extraction (by /u/USERNAME)",
                      client_id="14 digit code", client_secret="27 digit code",
                      username="user_name", password="password")
sub = reddit.subreddit('gonewild').hot(limit=10)
subreddit = list(sub)

for post in subreddit[2:]:
  submission = reddit.submission(id=post.id)
  submission.comments.replace_more(limit=0)
  for comment in submission.comments.list():
      p = comment.body.encode('utf-8')
      p = str(p.decode(encoding = 'utf-8'))
#    if type(i)!= 'bytes':
      f.write(deEmojify(p))
      f.write('\n')