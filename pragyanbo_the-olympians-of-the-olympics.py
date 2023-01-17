import numpy as np
import pandas as pd 
df = pd.read_csv('../input/athlete_events.csv')
df.columns
regions = pd.read_csv('../input/noc_regions.csv')
regions.head(2)
df.shape
df.columns
df.ID.value_counts().head(2)
robert = df[df.ID==77710].reset_index(drop=True)
robert.head(10)
df = df.drop_duplicates(['ID','NOC','Age','City','Season','Event','Team'])
df.shape
df['Medal'].unique()
df['Medal'].fillna(0,inplace=True)
df.groupby(['Medal','ID','Name'])['Team'].count().reset_index().rename(columns={'Team':'Number of medals'}).sort_values('Number of medals',ascending=False).drop_duplicates('Medal')
