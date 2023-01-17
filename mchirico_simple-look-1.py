# Import pandas, a data processing and CSV file I/O library
import pandas as pd
import numpy as np
import datetime

import nltk
import random


# Seaborn, a Python graphing library
import warnings # current version of seaborn generates a bunch of warnings. Ignore
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)


dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')


# There are just 4 files

p = pd.read_csv("../input/post.csv",
     header=0,names=['gid','pid', 'id', 'name', 'timeStamp', 'shares', 'url', 'msg', 'likes'],
     dtype={'gid':str,'pid':str,'id':str,'name':str,'timeStamp':datetime.datetime,'shares':str,
     'url':str,'msg':str,'likes':str}, 
     parse_dates=['timeStamp'],date_parser=dateparse)

c = pd.read_csv("../input/comment.csv",
     header=0,names=['gid','pid', 'cid', 'timeStamp', 'id', 'name', 'rid', 'msg'],
     dtype={'gid':str,'pid':str,'cid':str,'timeStamp':datetime.datetime,'id':str,
        'name':str,'rid':str,'msg':str}, 
     parse_dates=['timeStamp'],date_parser=dateparse)


l = pd.read_csv("../input/like.csv",
                header=0,names=['gid','pid', 'cid', 'response', 'id', 'name'],
                dtype={'gid':str,'pid':str,'cid':str,'response':str,
                        'name':str})

mem = pd.read_csv("../input/member.csv",
                header=0,names=['gid','id', 'name', 'url'],
                dtype={'gid':str,'id':str,'name':str,'url':str})
                
# There are 3 Facebook groups
#    25160801076        Unofficial Cheltenham Township
#    117291968282998    Elkins Park Happenings!
#    1443890352589739   Free Speech

# Take one group, "Unofficial Cheltenham Township" and display a few records.
# These are the main posts.
p[p['gid']=='25160801076'].sort_values(by='timeStamp',ascending=False).head(12)
# Comments are tied to main posts, but just take 
# a quick look at a few comments.  Later join pid
# from comments to posts.
c.sort_values(by='timeStamp',ascending=False).head(7)
# Do a search on a main post
# Sewer I/I issues are a big deal for this township.
#
#  Search for all Main posts with the word "Sewer", that has over 30 likes
#  and over 20 shares.


p['likes'].fillna(0, inplace=True)
p['likes']=p['likes'].astype(float)
p['shares'].fillna(0, inplace=True)
p['shares']=p['shares'].astype(float)
p[(p['msg'].astype(str).str.contains('Sewer')) & 
  (p['likes'] > 20) & (p['shares'] > 20)  ].head(5) 
# pid '25160801076_10153922841366077' has a lot of likes and shares
# These groups are public, so the url should be a way to validate
# the data in the post:  
#      https://www.facebook.com/groups/25160801076/permalink/10153922841366077/
#
#  Following comments can be shown here.
c[c['pid']=='25160801076_10153922841366077'].sort_values(by='timeStamp',ascending=True) 
