import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

# This script identifies which communication styles receive highest ranks
# For illustration purposes I defined 3 styles such as Passive, Assertive and Aggressive
# The list of key words must of course be extended

sql_conn = sqlite3.connect('../input/database.sqlite')

df = pd.read_sql("SELECT * FROM May2015 WHERE body != '[deleted]' LIMIT 10000", sql_conn)

