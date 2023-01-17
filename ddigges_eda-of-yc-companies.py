import matplotlib.pyplot as plt

import numpy as np 

import pandas as pd

plt.style.use('fivethirtyeight')
df = pd.read_csv("../input/companies.csv")
df.dtypes
# Create a new categorical variable with the batch category, either 's' or 'w'

df.batch_type  = df.batch.apply(lambda x: x[:1]).astype('category')
df.head()
plot = df.name.groupby(df.year).count().plot.line()
plot = df.name.groupby(df.batch_type).count().plot.bar()
df.name.groupby([df.year, df.batch_type]).count().unstack().plot.bar()
df.name.groupby(df.vertical).count().sort_values().plot.bar()
df.name.groupby([df.vertical, df.batch_type]).count().unstack().plot.bar()
df.name.groupby([df.year, df.vertical]).count().unstack().plot.line()

plt.figure(figsize=(30, 20))