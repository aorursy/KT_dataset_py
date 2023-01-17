import pandas as pd

df = pd.read_csv("../input/autompg-dataset/auto-mpg.csv",na_values='?')

df.describe()

df.isnull().any()
horse_med=df['horsepower'].median()

print(horse_med)

df['horsepower'].fillna(horse_med,inplace=True)
df.isnull().any()
df.boxplot(column=['horsepower'])
df.horsepower.quantile([0.25,0.5,0.75])
