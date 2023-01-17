# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

sns.set(style="white", color_codes=True)

from wordcloud import WordCloud

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Read data

df = pd.read_csv('../input/startup_funding.csv')
df.head()
df.isnull().sum(axis=0)
df.duplicated().sum()
df.dtypes
df['Amount in USD'].unique()
df["Amount in USD"] = df['Amount in USD'].str.replace(',','').str.extract('(^\d*)')

df["Amount in USD"] = df['Amount in USD'].replace('',np.nan)

df['Amount in USD'] = df['Amount in USD'].astype(float)

# Replace with mean

df['Amount in USD'].fillna(df['Amount in USD'].mean(),inplace=True)
import seaborn as sns

corrmat = df.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, square=True);
import seaborn as sns

sns.distplot(df['Amount in USD'])
plt.scatter(range(len(df["Amount in USD"])),np.sort(df["Amount in USD"].values))

plt.xlabel("index")

plt.ylabel("Funding in USD")

plt.show()
df['Industry Vertical'].value_counts().head(10).plot.barh()
df['SubVertical'].value_counts().head(10).plot.barh()
df['Date ddmmyyyy'].value_counts().head(10).plot.barh()
sns.violinplot(x=df['Industry Vertical'][df['Industry Vertical'] == 'Consumer Internet'], y="Amount in USD", data=df);
# Frequency of each category

df['InvestmentnType'].value_counts()
df['InvestmentnType'][df['InvestmentnType'] == 'SeedFunding'] = 'Seed Funding'

df['InvestmentnType'][df['InvestmentnType'] == 'Crowd funding'] = 'Crowd Funding'

df['InvestmentnType'][df['InvestmentnType'] == 'PrivateEquity'] = 'Private Equity'
df['InvestmentnType'].value_counts().head().plot.bar()
df['City  Location'].value_counts().head(10).plot.bar()
names = df["Investorsxe2x80x99 Name"][~pd.isnull(df["Investorsxe2x80x99 Name"])]
wordcloud = WordCloud(max_font_size=50, width=600, height=300).generate(' '.join(names))
plt.figure(figsize=(12,6))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")