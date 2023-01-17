import pandas as pd
df = pd.read_csv("../input/melbourne-uv-2019/uv-melbourne-2019.csv")
df
df.info()
df.sample()
df.describe()
df.describe().T
df.isna().sum()
df.isnull().sum()
df.columns
df['UV_Index'].value_counts()