import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import tensorflow as tf

color = sns.color_palette()



%matplotlib inline



pd.options.mode.chained_assignment = None

pd.options.display.max_columns = 999
df = pd.read_csv("../input/startup_funding.csv")

df.shape
df.head()
### Some more fixes in the data format. Will try to fix in the input file in next version #

df['Date'][df['Date']=='12/05.2015'] = '12/05/2015'

df['Date'][df['Date']=='13/04.2015'] = '13/04/2015'

df['Date'][df['Date']=='15/01.2015'] = '15/01/2015'

df['Date'][df['Date']=='22/01//2015'] = '22/01/2015'

df["yearmonth"] = (pd.to_datetime(df['Date'],format='%d/%m/%Y').dt.year*100)+(pd.to_datetime(df['Date'],format='%d/%m/%Y').dt.month)



cnt_srs = df['yearmonth'].value_counts()

plt.figure(figsize=(12,6))

sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[1])

plt.xticks(rotation='vertical')

plt.xlabel('Year-Month of transaction', fontsize=12)

plt.ylabel('Number of fundings made', fontsize=12)

plt.title("Year-Month Distribution", fontsize=16)

plt.show()
AmountInUSD = df["AmountInUSD"].apply(lambda x: float(str(x).replace(",","")))

AmountInUSD = AmountInUSD[~np.isnan(AmountInUSD)]

plt.figure(figsize=(8,6))

plt.scatter(range(len(AmountInUSD)), np.sort(AmountInUSD.values), color=color[4])

plt.xlabel('index', fontsize=12)

plt.ylabel('Funding value in USD', fontsize=12)

plt.show()
ulimit = np.percentile(AmountInUSD.values, 99)

llimit = np.percentile(AmountInUSD.values, 1)

AmountInUSD[AmountInUSD>ulimit] = ulimit

AmountInUSD[AmountInUSD<llimit] = llimit



plt.figure(figsize=(12,8))

sns.distplot(np.log(AmountInUSD.values), bins=50, kde=False, color=color[2])

plt.xlabel('log of Amount in USD', fontsize=12)

plt.title("Log histogram of investment amount in USD", fontsize=16)

plt.show()
df['InvestmentType'][df['InvestmentType']=='SeedFunding'] = 'Seed Funding'

df['InvestmentType'][df['InvestmentType']=='PrivateEquity'] = 'Private Equity'

df['InvestmentType'][df['InvestmentType']=='Crowd funding'] = 'Crowd Funding'



cnt_srs = df['InvestmentType'].value_counts()

plt.figure(figsize=(12,6))

sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[3])

plt.xticks(rotation='vertical')

plt.xlabel('Investment Type', fontsize=12)

plt.ylabel('Number of fundings made', fontsize=12)

plt.title("Type of Investment made", fontsize=16)

plt.show()
cnt_srs = df['CityLocation'].value_counts()[:10]

plt.figure(figsize=(12,6))

sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[1])

plt.xticks(rotation='vertical')

plt.xlabel('City or Location of the startup', fontsize=12)

plt.ylabel('Number of fundings made', fontsize=12)

plt.title("Location of startups with funding", fontsize=16)

plt.show()
cnt_srs = df['IndustryVertical'].value_counts()[:10]

plt.figure(figsize=(12,6))

sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[5])

plt.xticks(rotation='vertical')

plt.xlabel('Industry vertical of the startup', fontsize=12)

plt.ylabel('Number of fundings made', fontsize=12)

plt.title("Industry verticals of startups with funding", fontsize=16)

plt.show()
from wordcloud import WordCloud



inv_names = []

for invs in df['InvestorsName']:

    for inv in str(invs).split(","):

        if inv != "":

            inv_names.append(inv.strip().lower().replace(" ","_"))



# wordcloud for display address

plt.figure(figsize=(12,6))

wordcloud = WordCloud(background_color='black', width=600, height=300, max_font_size=50, max_words=40).generate(" ".join(inv_names))

wordcloud.recolor(random_state=0)

plt.imshow(wordcloud)

plt.title("Wordcloud for Investor Names", fontsize=30)

plt.axis("off")

plt.show()
cnt_srs = pd.Series(inv_names).value_counts()[:10]

plt.figure(figsize=(12,6))

sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[5])

plt.xticks(rotation='vertical')

plt.xlabel('Investor Names', fontsize=12)

plt.ylabel('Number of fundings made', fontsize=12)

plt.title("Top Investors in Indian Startups", fontsize=16)

plt.show()