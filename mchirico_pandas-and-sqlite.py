import pandas as pd

import sqlite3





import warnings # current version of seaborn generates a bunch of warnings that we'll ignore

warnings.filterwarnings("ignore")

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="white", color_codes=True)

con = sqlite3.connect('../input/database.sqlite')

# There are 4 tables

post = pd.read_sql_query("SELECT * FROM post", con)

comment = pd.read_sql_query("SELECT * FROM comment", con)

like = pd.read_sql_query("SELECT * FROM like", con)



# We don't want url. That just displays the image for the person.

# Also need a column name change.

rmember = pd.read_sql_query("SELECT distinct id as rid, name  rname FROM member", con)



# We'll update comment to add the response name (rname) to comment

comment=pd.merge(comment, rmember, left_on='rid', right_on='rid',how='left')
# Quick look at new comment

comment.head(4)
# This Dataset has 3 Facebook Groups

# 117291968282998  Elkins Park Happenings          EPH

# 25160801076      Unofficial Cheltenham Township  UCT

# 1443890352589739 Free Speech Zone                FSZ

#



#   Took the idea below from (Eugenia Uchaeva)

 

comment.gid = comment.gid.map({'117291968282998': 'EPH', '25160801076': 'UCT', '1443890352589739': 'FSZ'})

comment["gid"].value_counts()
# Let's see how many comments for just one group

# Note: comment.rid == ''  Otherwise, you'll pick up a reply to a comment

comment[( comment.gid == 'EPH') &  (comment.rid=='')]["name"].value_counts().head(10)
# Let's see how many response to comments

# Note: comment.rid != ''  comments go 2 levels deep.

comment[( comment.gid == 'EPH') &  (comment.rid != '')]["rname"].value_counts().head(10)