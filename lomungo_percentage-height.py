import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3

connection = sqlite3.connect("../input/database.sqlite")
df = pd.read_sql_query("SELECT * from Player", connection)
plt.hist(df.height)
ratio = df[df.height > 195].count() / df.count() * 100
print('Percentage of Players > 195 cm in Fifa League Seasons 2008 - 2016: %s' % ratio.height.round(2))

