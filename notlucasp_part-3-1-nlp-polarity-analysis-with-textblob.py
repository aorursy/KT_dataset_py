from part1_cleaning import *

df1, df2, df3 = get_clean_data()
from textblob import TextBlob



polarities = []

for value in df1["Headlines"].values:

    wiki = TextBlob(value)

    polarities.append(wiki.sentiment.polarity)
polarities[0:10]
df1['tb_hl_polarity'] = polarities

df1
polarities = []

for value in df1["Description"].values:

    wiki = TextBlob(value)

    polarities.append(wiki.sentiment.polarity)
polarities[0:10]
df1['tb_ds_polarity'] = polarities

df1
import matplotlib.pyplot as plt

plt.scatter(df1['tb_hl_polarity'].values, df1['tb_ds_polarity'].values)

plt.show()
polarities = []

for value in df2["Headlines"].values:

    wiki = TextBlob(value)

    polarities.append(wiki.sentiment.polarity)
df2['tb_hl_polarity'] = polarities

df2
polarities = []

for value in df2["Description"].values:

    wiki = TextBlob(value)

    polarities.append(wiki.sentiment.polarity)
df2['tb_ds_polarity'] = polarities

df2
plt.scatter(df2['tb_hl_polarity'].values, df2['tb_ds_polarity'].values)

plt.show()
polarities = []

for value in df3["Headlines"].values:

    wiki = TextBlob(value)

    polarities.append(wiki.sentiment.polarity)
df3['tb_hl_polarity'] = polarities

df3
plt.hist(df3['tb_hl_polarity'], bins = 50)

plt.show()