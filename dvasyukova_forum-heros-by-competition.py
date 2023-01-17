from IPython.display import HTML

import pandas as pd

import re

import sqlite3



con = sqlite3.connect('../input/database.sqlite')



df = pd.read_sql_query("""

SELECT SUM(m.Score) Score,

       COUNT(DISTINCT m.Id) NumPosts,

       f.Id ForumId,

       u.Id UserId,

       u.UserName,

       u.DisplayName,

       f.Name ForumName

FROM ForumMessages m

INNER JOIN ForumTopics t on m.ForumTopicId = t.Id

INNER JOIN Forums f on t.ForumId = f.Id

INNER JOIN Users u on m.AuthorUserId = u.Id

WHERE f.ParentForumId = 8

GROUP BY f.Id, u.Id

ORDER BY f.Id DESC, Score DESC

""", con)



best = df.groupby('ForumId',sort=False).apply(pd.DataFrame.head,3).reset_index(drop=True)

best['User'] = "<a href='https://www.kaggle.com/u/" + best["UserId"].astype(str) + "'>" + best["DisplayName"] + "</a>"

HTML(best.set_index(['ForumName','User'])[['Score','NumPosts']].head(99).to_html(escape=False))