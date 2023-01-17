import pandas as pd
import pandas_profiling as pdp
df = pd.read_csv('train.csv')
pdp.ProfileReport(df)