import pandas as pd

import matplotlib.pyplot as plt
df = pd.read_csv("../input/random-time-series.csv")
print(df.head())
df.plot(x='date', y='value');
df['rank'] = (df['date'].rank(ascending=False) - 1).astype('int64')
print(df.head())
time_decay_const = 0.99



df['time_decay_value'] = df['value'] * pow(time_decay_const, df['rank'])
print(df.head())
df.plot(x='date', y='time_decay_value');