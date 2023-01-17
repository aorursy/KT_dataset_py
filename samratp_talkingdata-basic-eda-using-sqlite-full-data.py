import pandas as pd

import numpy as np

import sqlite3

import zipfile

import subprocess

import gc
df_train_sample = pd.read_csv('../input/train.csv', nrows=10000) #10k

df_train_sample.info()
df_train_sample.head()
df_test_sample = pd.read_csv('../input/test.csv', nrows=10000) #10k

df_test_sample.info()
df_test_sample.head()
con = sqlite3.connect("talkingdata_test.db")  # Opens file if exists, else creates file

cur = con.cursor()  # This object lets us actually send messages to our DB and receive results
sql = "SELECT sql FROM sqlite_master WHERE name='test_data'"

cur.execute(sql)



if not cur.fetchall():

    # In the below call you can use the actual test.csv.zip file when running it on a local machine

    for chunk in pd.read_csv("../input/test_supplement.csv", nrows=10000, chunksize=500):

        chunk.to_sql(name="test_data", con=con, if_exists="append", index=False)  #"name" is name of table

        gc.collect()
sql = "SELECT sql FROM sqlite_master WHERE name='test_data'"

cur.execute(sql)

cur.fetchall()
sql = "select count(*) from test_data"

cur.execute(sql)

cur.fetchall()
sql = "select * from test_data limit 10"

cur.execute(sql)

cur.fetchall()
con = sqlite3.connect("talkingdata_train.db")  # Opens file if exists, else creates file

cur = con.cursor()  # This object lets us actually send messages to our DB and receive results
sql = "SELECT sql FROM sqlite_master WHERE name='train_data'"

cur.execute(sql)



if not cur.fetchall():

    # In the below call you can use the actual train.csv.zip file when running it on a local machine

    for chunk in pd.read_csv("../input/train_sample.csv", nrows= 10000, chunksize=500):

        chunk.to_sql(name="train_data", con=con, if_exists="append", index=False)  #"name" is name of table

        gc.collect()
sql = "select count(*) from train_data"

cur.execute(sql)

cur.fetchall()
sql = "SELECT sql FROM sqlite_master WHERE name='train_data'"

cur.execute(sql)

cur.fetchall()
sql = "select * from train_data limit 10"

cur.execute(sql)

cur.fetchall()
sql = "select min(ip), max(ip) from train_data"

cur.execute(sql)

cur.fetchall()
sql = "select min(app), max(app) from train_data"

cur.execute(sql)

cur.fetchall()
sql = "select min(device), max(device) from train_data"

cur.execute(sql)

cur.fetchall()
sql = "select min(os), max(os) from train_data"

cur.execute(sql)

cur.fetchall()
sql = "select min(channel), max(channel) from train_data"

cur.execute(sql)

cur.fetchall()
sql = "select min(is_attributed), max(is_attributed) from train_data"

cur.execute(sql)

cur.fetchall()
sql = "select count(*) from train_data where is_attributed=1"

cur.execute(sql)

cur.fetchall()
sql = "select ip, count(ip) from train_data group by ip order by 2 desc limit 25"

cur.execute(sql)

cur.fetchall()
sql = "select ip, count(ip) from train_data where is_attributed=1 group by ip order by 2 desc limit 20"

cur.execute(sql)

cur.fetchall()