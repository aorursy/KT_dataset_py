%matplotlib inline
import sqlite3, pandas as pd, numpy as np
# Look at earnings data overall

# Read data from the SQLite datbase
con = sqlite3.connect('../input/database.sqlite')
earnings = pd.read_sql_query("""
SELECT opeid, count_wne_p10, count_nwne_p10, mn_earn_wne_p10, md_earn_wne_p10, pct10_earn_wne_p10, pct25_earn_wne_p10, pct75_earn_wne_p10, pct90_earn_wne_p10
FROM Scorecard
WHERE count_wne_p10 IS NOT NULL""", con)

earnings.head()

for col in earnings.columns:
    earnings.loc[earnings[col] == 'PrivacySuppressed', col] = None
np.sum(earnings.mn_earn_wne_p10.isnull())
earnings.describe()
from matplotlib import pyplot as plt
import seaborn
seaborn.set()
earnings.count_nwne_p10.hist();
earnings.shape
plt.plot(earnings.count_wne_p10, earnings.md_earn_wne_p10, 'o');