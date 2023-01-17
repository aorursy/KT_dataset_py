import os, sys

import csv

#http://blog.revolutionanalytics.com/2016/01/pipelining-r-python.html

%load_ext rpy2.ipython

import numpy, scipy, pandas

from rpy2.robjects import r, pandas2ri

pandas2ri.activate()

from collections import Counter

import datetime

import sqlite3

import zipfile

import nltk, re

%matplotlib inline

import matplotlib.pyplot as plt

plt.style.use('ggplot')



conn = sqlite3.connect('../input/database.sqlite')

c = conn.cursor()



import pandas.io.sql as psql

sql = "SELECT SUBSTR(MetadataDateSent,1,7) as SentFOIA, ExtractedDateSent as Sent, ExtractedBodyText as EmailBody FROM Emails e INNER JOIN Persons p ON e.SenderPersonId=P.Id WHERE p.Name='Hillary Clinton'  AND ExtractedBodyText != '' AND MetadataDateSent != '' ORDER BY MetadataDateSent"

df = psql.frame_query(sql, conn)

#df