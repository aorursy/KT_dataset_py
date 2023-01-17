import pandas as pd

pd.read_csv("../input/scrubbed.csv")
pd.DataFrame.describe(pd.read_csv("../input/scrubbed.csv"))