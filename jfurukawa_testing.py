import re 
import sqlite3 as sql
import matplotlib.pyplot as mpl
import numpy as np
import nltk
from pandas import *
from collections import Counter 


reddDB = sql.connect('../input/database.sqlite')

sql = "SELECT *      \
    FROM May2015                \
    LIMIT 10";

df = pandas.io.sql.read_frame(sql,reddDB)

print (df.ndim)

for rec in reddDB.execute(sql):
    print (str(rec))

