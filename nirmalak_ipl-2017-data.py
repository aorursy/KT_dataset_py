%matplotlib inline

import pandas as pd
deliveries = pd.read_csv('../input/deliveries.csv')

deliveries.head(5)
deliveries.columns
deliveries.groupby('batting_team').agg(sum)['batsman_runs'].plot(kind='bar')