import pandas as pd

import numpy as np

import datetime

import sqlite3





import warnings

warnings.filterwarnings("ignore")

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="white", color_codes=True)





dateparse = lambda x: datetime.datetime.strptime(x,'%Y-%m-%d')



# Read data for Pandas

d=pd.read_csv("../input/debate.csv",encoding = "ISO-8859-1")







# SQLITE

c = sqlite3.connect(":memory:")

d.to_sql("debate",c)

df = pd.read_sql("SELECT  Speaker,Date, COUNT(*) as Count  FROM debate GROUP BY Speaker,Date ORDER BY Speaker,Date DESC;", c)
d.head()
df.head(10)