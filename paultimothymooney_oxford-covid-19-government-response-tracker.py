import pandas as pd

import seaborn as sns

df = pd.read_excel('/kaggle/input/oxford-covid19-government-response-tracker/OxCGRT_Download_latest_data.xlsx')
df.sort_values('Date',ascending=False).head(10)
sns.heatmap(df.isnull(), cbar=False)
df.columns.values
df.shape