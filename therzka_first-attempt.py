# The data comes both as CSV files and a SQLite database

import pandas as pd
import matplotlib.pyplot as plt
import sqlite3

# You can read in the SQLite datbase like this
#import sqlite3
#con = sqlite3.connect('../input/database.sqlite')
#sample = pd.read_sql_query("""
#SELECT *
#FROM Teams
#LIMIT 10""", con)
#print(sample)

# You can read a CSV file like this
competitions = pd.read_csv("../input/Competitions.csv")
descs = competitions["SolutionNumRows"]
submissions = competitions["NumScoredSubmissions"]

df = pd.DataFrame({"SolutionNumRows": submissions, "descriptions": descs})
print(df)

# It's yours to take from here!
