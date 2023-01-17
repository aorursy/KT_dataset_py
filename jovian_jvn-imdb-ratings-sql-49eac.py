# Initialize OK

from client.api.notebook import Notebook

ok = Notebook('hw3.ok')
import numpy as np

import pandas as pd

import sqlite3
import matplotlib.pyplot as plt

import plotly.offline as py

import cufflinks as cf
from ds100_utils import run_sql

help(run_sql)
for _, row in run_sql("SELECT sql FROM sqlite_master WHERE type='table' ").iterrows():

    print(row['sql'], '\n')
run_sql("""

    SELECT * FROM names LIMIT 10

""")
run_sql("""

    SELECT * FROM titles LIMIT 10

""")
run_sql("""

    SELECT * FROM roles LIMIT 10

""")
run_sql("""

    SELECT * FROM ratings LIMIT 10

""")
oldestMovieQuery = """SELECT start_year, type, title FROM titles WHERE start_year NOT IN ('NULL') ORDER BY start_year, title ASC LIMIT 10"""

oldestMovieDf = run_sql(oldestMovieQuery)

oldestMovieDf
ok.grade("q1");
yearDistQuery = """SELECT start_year, COUNT(*) AS total FROM titles WHERE start_year NOT IN ('NULL') GROUP BY start_year"""

yearDistDf = run_sql(yearDistQuery)

yearDistDf
ok.grade("q2");
yearDistDf.iplot(kind="bar", x="start_year", y="total", 

                 xTitle="Start Year", yTitle="Count", asFigure=True)
typeQuery = """SELECT DISTINCT(type) FROM titles"""

typeDf = run_sql(typeQuery)

typeDf
ok.grade("q3");
roleCategoriesQuery = """SELECT category, COUNT(*) as total FROM roles GROUP BY category ORDER BY total DESC"""

roleCategoriesDf = run_sql(roleCategoriesQuery)

roleCategoriesDf
ok.grade("q4");
roleCategoriesDf.iplot(kind="barh", x ="category", y = "total", xTitle="Count", asFigure=True)
prolificPerformersQuery = """SELECT name, COUNT(*) as total FROM names n INNER JOIN roles r ON n.nid = r.nid INNER JOIN titles t ON t.tid = r.tid WHERE t.type = 'movie' AND r.category IN ('actor', 'actress') GROUP BY n.nid ORDER BY total DESC, name ASC LIMIT 10"""

prolificPerformersDf = run_sql(prolificPerformersQuery)

prolificPerformersDf
ok.grade("q5");
missingRatingsQuery = """SELECT CASE WHEN num_votes IS NOT NULL THEN 'yes' ELSE 'no' END as has_rating, COUNT(*) as total FROM titles t LEFT JOIN ratings r ON t.tid = r.tid WHERE t.type = 'movie' GROUP by has_rating"""

missingRatingsDf = run_sql(missingRatingsQuery)

missingRatingsDf
ok.grade("q6");
popularVotesQuery = """SELECT title, num_votes, avg_rating FROM titles t INNER JOIN ratings r ON t.tid = r.tid GROUP BY t.title ORDER BY r.num_votes DESC LIMIT 10"""

popularVotesDf = run_sql(popularVotesQuery)

popularVotesDf
ok.grade("q7");
runtimeRatingsQuery = """SELECT ROUND(runtime_minutes / 10.0 + 0.5) * 10 as runtime_bin, avg(avg_rating) as avg_rating, avg(num_votes) as avg_num_votes, COUNT(*) as total FROM titles t INNER JOIN ratings r ON t.tid = r.tid WHERE r.num_votes >= 10000 AND t.type = 'movie' GROUP BY runtime_bin"""

runtimeRatingsDf = run_sql(runtimeRatingsQuery)

runtimeRatingsDf
ok.grade("q8");
runtimeRatingsDf.iplot(x="runtime_bin", y="avg_rating", asFigure=True)
runtimeRatingsDf.iplot(x="runtime_bin", y="avg_num_votes", asFigure=True)
runtimeRatingsDf.iplot(kind="bar", x="runtime_bin", y="total", asFigure=True)
topRatedPerformerQuery = """SELECT name, sum(avg_rating*num_votes)/sum(num_votes) as avg_rating FROM names n INNER JOIN roles x ON n.nid = x.nid INNER JOIN ratings r ON x.tid = r.tid INNER JOIN titles t ON r.tid = t.tid WHERE t.type = 'movie' AND num_votes >= 1000 AND x.category IN ('actor', 'actress') GROUP BY name HAVING COUNT(*) >= 20 ORDER BY avg_rating DESC, name ASC LIMIT 10"""

topRatedPerformerDf = run_sql(topRatedPerformerQuery)

topRatedPerformerDf
ok.grade("q9");
# Save your notebook first, then run this cell to submit.

ok.submit()