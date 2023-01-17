import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



df = pd.read_csv("../input/elnino.csv")



# Remove extra space in columns

df.columns = [col.strip() for col in df.columns]



df.head()
# Air Temp summary statistics

df['Air Temp'] = pd.to_numeric(df['Air Temp'], errors='coerce')

df['Air Temp'].describe()
# Sea Surface Temp summary statistics

df['Sea Surface Temp'] = pd.to_numeric(df['Sea Surface Temp'], errors='coerce')

df['Sea Surface Temp'].describe()
sns.jointplot(x="Air Temp", y="Sea Surface Temp", data=df, size=7)