import pandas as pd

import sqlite3



con = sqlite3.connect("../input/database.sqlite")

pd.read_sql_query("SELECT * FROM Users LIMIT 100", con)