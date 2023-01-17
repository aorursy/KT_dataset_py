import pandas as pd

df = pd.read_csv("../input/ieeefrauddetectionleaderboardcsv/ieee-fraud-detection.csv")
df.sort_values(by=['Rank']).head(3)
df[df['Medal']=='gold'].sort_values(by=['Rank']).head(21).sort_values(by=['PB_change'], ascending=False)
df[df['Medal']=='silver'].sort_values(by=['Rank']).head(299).sort_values(by=['PB_change'], ascending=False)
df[df['Medal']=='bronze'].sort_values(by=['Rank']).head(322).sort_values(by=['PB_change'], ascending=False)
import numpy as np

df [df['PB_change'] != np.nan].sort_values(by=['Rank']).sort_values(by=['PB_change'], ascending=False).head(50)
import numpy as np

df [df['PB_change'] != np.nan].sort_values(by=['Rank']).sort_values(by=['PB_change'], ascending=False).tail(50)