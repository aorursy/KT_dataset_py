import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../input/data.csv')
df.columns
df_made = df[['shot_made_flag','opponent']]
sorted = df_made.groupby(['opponent']).mean().sort(['shot_made_flag'])
sorted