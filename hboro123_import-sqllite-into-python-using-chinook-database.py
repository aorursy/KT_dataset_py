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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sqlite3

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



path = "../input/"  #Insert path here

database = path + 'Chinook_Sqlite.sqlite'
## creating a connection

conn = sqlite3.connect(database)



## importing tables 

tables = pd.read_sql("""SELECT *

                        FROM sqlite_master

                        WHERE type='table';""", conn)



tables
## Getting the primary key of the album table



album_primary_keys= pd.read_sql("""PRAGMA table_info(album);""",conn)

album_primary_keys



## so primary keys are albumid
## Extracting the album table

album = pd.read_sql("""SELECT *

                        FROM album

                        ;""", conn)

album
# number of rows in album column



count_rows= pd.read_sql("""SELECT COUNT(*)FROM album;""",conn)

count_rows
# Unique album



unique_ablum= pd.read_sql("""SELECT DISTINCT(Title) FROM album;""",conn)

unique_ablum
## checking null value is there or not



null_check_album= pd.read_sql(""" SELECT * FROM album WHERE title IS NULL;""",conn )

null_check_album
## checking longest name of album



longest_name_album= pd.read_sql("""SELECT title,max(LENGTH(title)) FROM album;""",conn)

longest_name_album
## order by alphabetical order of title of album



order_title_desc_alphabetical= pd.read_sql("""SELECT AlbumId,title,length(title),ArtistId from album order by title desc;""",conn)

order_title_desc_alphabetical

## info about the three tables



album_info= pd.read_sql("""PRAGMA table_info(album);""",conn)

print('album: ',album_info)



print('\n')

artist_info= pd.read_sql("""PRAGMA table_info(artist);""",conn)

print('artist: ',artist_info)



print('\n')

genre_info= pd.read_sql("""PRAGMA table_info(genre);""",conn)

print('genre: ',genre_info)





print('\n')

customer_info= pd.read_sql("""PRAGMA table_info(customer);""",conn)

print('customer: ',customer_info)



print("\n")



employee_info= pd.read_sql("""PRAGMA table_info(employee);""",conn)

print('employee:',employee_info)



print("\n")

invoice_info= pd.read_sql("""PRAGMA table_info(Invoice);""",conn)

print('invoice:',invoice_info)

## join album and artist as both have common column albumid for join



artist_album_join=pd.read_sql("""SELECT album.AlbumId,NAME,album.Title FROM ARTIST  JOIN ALBUM ON ALBUM.ARTISTID= ARTIST.ARTISTID;""",conn)

artist_album_join
## join customer and invoice as both have common column customerid for join



customer_invoice_join=pd.read_sql("""SELECT customer.customerid,InvoiceDate,Total  FROM invoice  JOIN customer ON customer.customerid= invoice.customerid;""",conn)

customer_invoice_join.info()
## change invoice date to datetime object



customer_invoice_join['InvoiceDate']=pd.to_datetime(customer_invoice_join['InvoiceDate'])





## Timeseries plot



sns.lineplot(x='InvoiceDate',y='Total',data=customer_invoice_join)



plt.show()
## data distribution of total



sns.distplot(customer_invoice_join['Total'], kde=False, color='red', bins=10)

plt.show()