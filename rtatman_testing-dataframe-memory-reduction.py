import pandas as pd

import fail_safe_parallel_memory_reduction as reducing
# load in the data...

df = pd.read_csv("../input/international-football-results-from-1872-to-2017/results.csv")



df.head()
# ...and make it smaller!

df = reducing.Reducer().reduce(df)
# still the same data, though :)

df.head()