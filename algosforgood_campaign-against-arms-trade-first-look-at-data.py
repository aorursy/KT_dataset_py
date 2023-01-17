import numpy as np 
import pandas as pd
import sqlite3
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# Read sqlite query results into a pandas DataFrame
connection = sqlite3.connect("../input/database.sqlite")

#The below lines gives an error
dataframe = pd.read_sql_query("SELECT * from criteria", connection)

# close sqlite connection
connection.close()
