import csv

import pandas as pd

import sqlite3



# Connect to database

db = sqlite3.connect("../input/cord-19-analysis-with-sentence-embeddings/cord19q/articles.sqlite")



# Articles

articles = pd.read_sql_query("select id, title, design, size, sample, method, reference from articles where tags is not null order by design, published", db)

articles.to_csv("metadata_study.csv", index=False, header=True, quoting=csv.QUOTE_NONNUMERIC)