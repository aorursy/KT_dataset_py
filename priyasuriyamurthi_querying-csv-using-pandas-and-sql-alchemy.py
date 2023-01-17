import pandas as pd

who_suicide_statistics = pd.read_csv("../input/who-suicide-statistics/who_suicide_statistics.csv")
who_suicide_statistics.head()
from sqlalchemy import create_engine

engine = create_engine('sqlite://', echo=False)

who_suicide_statistics.to_sql("suicides", con=engine)
result = engine.execute("""SELECT * FROM suicides where country = 'Brazil' and year between 1985 and 1987

                        and sex = 'female' """)
dataframe = pd.DataFrame(result.fetchall())

dataframe.columns = result.keys()

print(dataframe)