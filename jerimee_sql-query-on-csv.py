import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sqlite3 as sql

import csv # what is this?



limitResults = 72

limitFormatTypes = True



dbName = "bookdepo.db"

conn = sql.connect(dbName) 

doesTableExist = "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='books';"

df = pd.read_sql(doesTableExist, conn)

tableExists = df.values.any()

if (tableExists != True):

    print("- - table does not exist yet")

    print("- - read csv")

    fileIn = "/kaggle/input/book-depository-dataset/dataset.csv"

    df = pd.read_csv(fileIn, thousands=None)

    print("- - remove commas from bestsellers col")

    # df["bestsellers-rank"] = pd.to_numeric(df["bestsellers-rank"])

    # df.astype({'bestsellers-rank': 'int32'})

    df['bestsellers-rank'] = df['bestsellers-rank'].str.replace(',', '')

    print("- - create 'books' table")

    df.to_sql('books', conn)

else:

    print("- - books table found in " + dbName)



limitString = str(limitResults)

col1 = '"authors", "bestsellers-rank", "categories", "description", "dimension-x"'

col2 = '"dimension-y", "dimension-z", "edition", "edition-statement", "for-ages"'

col3 = '"format", "id", "illustrations-note", "imprint", "index-date"'

col4 = '"isbn10", "isbn13", "lang", "publication-date", "publication-place"'

col5 = '"rating-avg", "rating-count", "title", "url", "weight"'

seperator = ', '

s = seperator

cols = col1 + s + col2 + s + col3 + s + col4 + s + col5

qFormatIn = ' AND format IN (1,2,3,13,15,26)'

queryStart = 'SELECT ' + cols + ' FROM books WHERE lang = "en" AND description IS NOT NULL'

# distinct ids

# queryStart = 'SELECT COUNT("id"), id FROM books WHERE lang = "en" AND description IS NOT NULL GROUP BY title'

if (limitResults > 0):

    limsql = ' LIMIT '

    queryEnd = limsql + limitString

else:

    limitString = ""

    queryEnd = ""



if (limitFormatTypes == True):

    query = queryStart + qFormatIn + queryEnd

else: 

    query = queryStart + queryEnd



print("- - running query")

print(query)

print(" ")

df = pd.read_sql(query, conn)



fileNameStart = "booksEng"

fileNameLimit = limitString

fileNameExt = ".csv"

fileName = fileNameStart + fileNameLimit + fileNameExt

fileOut = fileName



print("- - writing to file")

print(fileOut)

print(" ")

df.to_csv(fileOut,index=False,quoting=csv.QUOTE_ALL,line_terminator='\r\n')

print ("- - done")
import pandas as pd

import sqlite3 as sql

conn = sql.connect("mydb.db") 

doesTableExist = "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='books';"

df = pd.read_sql(doesTableExist, conn)

df.values.any()
#Show Columns

import pandas as pd

import sqlite3 as sql

conn = sql.connect("mydb.db")

q = "PRAGMA table_info(books)"

out = pd.read_sql(q, conn)

print(out)
df.dtypes