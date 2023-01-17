# The data comes as the raw data files, a transformed CSV file, and a SQLite database

import pandas as pd


# You can read a CSV file like this
scorecard = pd.read_csv("../input/Scorecard.csv")
#print(scorecard)

# It's yours to take from here!



scorecard.head()
colname = " "
for i in scorecard.columns:
    colname = colname + i + ","

print(colname)
scorecard.loc[:10, ['INSTNM']]

