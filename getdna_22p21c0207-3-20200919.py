import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
Data = pd.read_csv("../input/data-for-datavis/museum_visitors.csv", index_col="Date", parse_dates=True)
Data.head()
plt.figure(figsize=(10,6))

sns.lineplot(data=Data)
plt.figure(figsize=(10,6))

sns.barplot(x=Data.columns, y=Data.iloc[0])