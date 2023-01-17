import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

sql_conn = sqlite3.connect('../input/database.sqlite')

query = """SELECT candidate, count(candidate) as frequency
        FROM Sentiment 
        GROUP BY candidate
        ORDER BY frequency"""
candidate_mentions = pd.read_sql(query, sql_conn)
print(candidate_mentions)

