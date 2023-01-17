import pandas as pd 



import os

print(os.listdir("../input"))
df = pd.read_csv("../input/googleplaystore.csv")

df.head() # show only the first 5 results
# import sqlalchemy and create a sqlite engine

from sqlalchemy import create_engine

engine = create_engine('sqlite://', echo=False)



# export the dataframe as a table 'playstore' to the sqlite engine

df.to_sql("playstore", con=engine)
result = engine.execute("SELECT * FROM playstore")
dataframe = pd.DataFrame(result.fetchall())

dataframe.columns = result.keys()

dataframe.head() # only show first 5 results / tuples

# to get all results uncomment the next line

# dataframe 
query = engine.execute("SELECT App, Genres, Size FROM playstore WHERE Size > 10 AND Genres NOT LIKE 'Art & Design%'")
df = pd.DataFrame(query.fetchall())

df.columns = query.keys()

df.head()