import pandas as pd

import matplotlib.pyplot as plt

import datetime



df = pd.read_csv("../input/vgsale_1.csv")

df.head()



df_bf2000 = df[df['Year'] <= 2000]

df_af2000 = df[df['Year'] > 2000]

df.head()







df_af2000.drop_duplicates(['Name'])

df_af2000.Genre.value_counts().plot.bar() #популярность жанров после 2000 года по кол-ву игр
df_af2000.groupby('Genre')['Global_Sales'].sum().plot.bar() #популярность жанров после 2000 года по продажам
df.groupby('Year')['Name'].count().plot.bar(figsize=(22, 7)) #общее число видеоигр в каждом году
dfTopPub = df.drop_duplicates(['Name'])

dfTopPub = dfTopPub.groupby(['Publisher'])['Name'].count().nlargest(3)

dfPf = df[df["Publisher"].isin(dfTopPub.index)]

dfPf  = dfPf.groupby(['Publisher','Platform'])['Name'].count() #общее число видеоигр в каждом году

dfPf  = dfPf.unstack(level=0)

dfPf.plot.bar(figsize=(22, 7), layout=(3, 2), stacked=True)
df_bf2000 = df_bf2000[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']].sum()

df_af2000 = df_af2000[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']].sum()



total = pd.DataFrame({"1980-2000": df_bf2000, "2000-2020": df_af2000})



total.plot.pie(subplots=True, legend=True, figsize=(13, 13))