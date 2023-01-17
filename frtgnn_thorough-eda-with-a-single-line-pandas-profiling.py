import pandas as pd 

import pandas_profiling
df = pd.read_csv('../input//titanic/train.csv')



report = pandas_profiling.ProfileReport(df)

report.to_file("report.html")



report