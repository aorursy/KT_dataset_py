import sqlite3

import os



#Export count

count = 500



# Remove existing db, if present

if os.path.exists("articles.sqlite"):

    os.remove("articles.sqlite")



# Create db and attach existing db

db = sqlite3.connect("articles.sqlite")

db.execute("ATTACH '../input/cord-19-etl/cord19q/articles.sqlite'as indb")

cur = db.cursor()



# Copy data

cur.execute("CREATE TABLE articles AS SELECT * FROM indb.articles WHERE tags IS NOT NULL ORDER BY RANDOM() LIMIT " + str(count))

cur.execute("CREATE TABLE sections AS SELECT * FROM indb.sections s WHERE s.article IN (SELECT id FROM articles)")



# Show counts

cur.execute("SELECT COUNT(*) FROM articles")

print("Total articles exported %d" % cur.fetchone())



cur.execute("SELECT COUNT(*) FROM sections")

print("Total sections exported %d" % cur.fetchone())



# Commit and close

db.commit()

db.close()