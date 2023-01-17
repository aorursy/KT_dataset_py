# Install cord19q project

!pip install git+https://github.com/neuml/cord19q
import sqlite3

import pandas as pd



# Connect to articles database

db = sqlite3.connect("../input/cord-19-analysis-with-sentence-embeddings/cord19q/articles.sqlite")



# Export last 200 records by published date to CSV

recent = pd.read_sql_query("SELECT title, published as Date, reference as URL, tags FROM articles order by published desc LIMIT 200", db)

recent.to_csv("cord-19_recent_200_date.csv", index=False)



# Export last 200 records by published date to CSV

recent = pd.read_sql_query("SELECT title, published as Date, reference as URL, tags FROM articles where published <= '2020-03-31' order by published desc LIMIT 200", db)

recent.to_csv("cord-19_recent_200_2020March.csv", index=False)



# Export last 200 records by id to CSV

recent = pd.read_sql_query("select title, published as Date, reference as URL, tags FROM articles a where id in " +

                           "(select distinct(article) from sections order by id desc LIMIT 200) order by (select max(id) from sections where article=a.id) desc", db)

recent.to_csv("cord-19_recent_200_id.csv", index=False)



import os

import shutil



import pandas as pd



from cord19q.models import Models

from cord19q.query import Query



if not os.path.exists("/tmp/cord19-300d.magnitude"):

    # Copy vectors locally for predictable performance

    shutil.copy("../input/cord19-fasttext-vectors/cord19-300d.magnitude", "/tmp")



embeddings, db = Models.load("../input/cord-19-analysis-with-sentence-embeddings/cord19q")

cur = db.cursor()



# Query embeddings index

rows = [(score, text) for uid, score, article, text in Query.search(embeddings, cur, "hypertension", 50)]



# Convert to dataframe and export

df = pd.DataFrame(rows, columns=["Score", "Text"])

df.to_csv("cord-19-top50-hypertension.csv", index=False)



Models.close(db)

import csv

import shutil

import sqlite3



import pandas as pd



db = sqlite3.connect("../input/cord-19-analysis-with-sentence-embeddings/cord19q/articles.sqlite")

cur = db.cursor()



# Output rows

rows = []



# Read training data, convert to features

with open("../input/cord19-study-design/design.csv", mode="r") as csvfile:

    for row in csv.DictReader(csvfile):

        uid = row["id"]

        label = row["label"]



        cur.execute("select id, title, published, reference from articles a where id=?", [uid])

        row = cur.fetchone()

        if row:

            rows.append((label,) + row)



df = pd.DataFrame(rows, columns=["Label", "Id", "Title", "Date", "URL"])

df.to_csv("cord-19-design.csv", index=False)



# Copy attribute.csv file over

shutil.copy("../input/cord19-study-design/attribute.csv", "cord-19-attribute.csv")