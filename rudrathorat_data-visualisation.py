import pandas as pd

df = pd.read_csv('../input/park-biodiversity/parks.csv', index_col=['Park Code'])
df.plot.bar()
df.plot.line()
df.plot.area()
df.plot.hist()
