import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from subprocess import check_output
from wordcloud import WordCloud, STOPWORDS
df_startups = pd.read_csv("../input/startup_funding.csv",index_col=0)
print(df_startups.shape)
print(df_startups.nunique())
print(df_startups.info())
print("NaN's description")
print(df_startups.isnull().sum())
df_startups.head()
df_startups.shape
df_startups["AmountInUSD"] = df_startups["AmountInUSD"].apply(lambda x: float(str(x).replace(",","")))
print("Min Amount")
print(df_startups["AmountInUSD"].min())
print("Mean Amount")
print(round(df_startups["AmountInUSD"].mean(),2))
print("Median Amount")
print(df_startups["AmountInUSD"].median())
print("Max Amount")
print(df_startups["AmountInUSD"].max())
print("Standard Deviation Amount")
print(round(df_startups["AmountInUSD"].std(),2))
#Let's create a new feature that is a Amount with log to better see the values distribuitions
df_startups['AmountInUSD_log'] = np.log(df_startups["AmountInUSD"] + 1)

plt.figure(figsize=(8,5))
sns.distplot(df_startups['AmountInUSD_log'].dropna())
plt.xlabel('log of Amount in USD', fontsize=12)
plt.title("Log Hist of investment in USD", fontsize=16)
plt.show()
InvestmentTypeVC = df_startups.InvestmentType.value_counts()
print("Description of Investiment Types: ")
print(InvestmentTypeVC)

plt.figure(figsize = (15,12))
plt.subplot(221)

g = sns.countplot(x="InvestmentType", data=df_startups)
g.set_xticklabels(g.get_xticklabels(),rotation=45)
g.set_title("Investiment Type count", fontsize=15)
g.set_xlabel("investment Types", fontsize=12)
g.set_ylabel("Count", fontsize=12)

plt.subplot(222)
g1 = sns.boxplot(x="InvestmentType", y="AmountInUSD_log", 
            data=df_startups)
g1.set_xticklabels(g.get_xticklabels(),rotation=45)
g1.set_title("Investment Types less than 1 bi", fontsize=15)
g1.set_xlabel("Investment Types", fontsize=12)
g1.set_ylabel("Amount(USD)", fontsize=12)

plt.show()
Investors = df_startups.InvestorsName.value_counts()

print("Description count of Investors")
print(Investors[:15])

plt.figure(figsize = (12,5))
g = sns.barplot(x=Investors.index[:20],y=Investors.values[:20])
g.set_xticklabels(g.get_xticklabels(),rotation=45)
g.set_title("Investors Name Count", fontsize=15)
g.set_xlabel("", fontsize=12)
g.set_ylabel("Count", fontsize=12)

plt.show()
location = df_startups['CityLocation'].value_counts()
print("Description count of Location")
print(location[:5])

plt.figure(figsize=(13,7))

plt.subplot(211)
sns.barplot(x=location.index[:20], y=location.values[:20])
plt.xticks(rotation=45)
plt.xlabel('City Location', fontsize=15)
plt.ylabel('City Count', fontsize=15)
plt.title("City Location Count ", fontsize=20)

plt.subplot(212)
g = sns.boxplot(x='CityLocation', y="AmountInUSD_log",
                data=df_startups[df_startups.CityLocation.isin(location[:15].index.values)])
g.set_xticklabels(g.get_xticklabels(),rotation=90)
g.set_title("City Localization by Amount LogDistribuition", fontsize=20)
g.set_xlabel("Top 15 Citys", fontsize=15)
g.set_ylabel("Amount (USD) - Log", fontsize=15)

plt.subplots_adjust(hspace = 0.65,top = 0.9)

plt.show()
industry = df_startups['IndustryVertical'].value_counts()[:20]

plt.figure(figsize=(12,6))
sns.barplot(x=industry.index, y=industry.values)
plt.xticks(rotation=45)
plt.xlabel("Industry's Name", fontsize=12)
plt.ylabel('Industry counting', fontsize=12)
plt.title("Count frequency of Industry Verical", fontsize=16)
plt.show()
cons_sub = df_startups['SubVertical'].value_counts()

plt.figure(figsize = (12,5))

g = sns.barplot(x=cons_sub.index[:20],y=cons_sub.values[:20])
g.set_xticklabels(g.get_xticklabels(),rotation=90)
g.set_title("Subvertical Count", fontsize=20)
g.set_xlabel("", fontsize=15)
g.set_ylabel("Count", fontsize=15)

plt.show()
tech_sub = df_startups[df_startups['IndustryVertical'] == 'Technology']['SubVertical'].value_counts()

plt.figure(figsize = (10,6))
g = sns.barplot(x=tech_sub.index[:20],y=tech_sub.values[:20])
g.set_xticklabels(g.get_xticklabels(),rotation=90)
g.set_title("Sub Category's by Technology", fontsize=15)
g.set_xlabel("", fontsize=12)
g.set_ylabel("Count", fontsize=12)
plt.show()
tech_sub = df_startups[df_startups['IndustryVertical'] == 'Ecommerce']['SubVertical'].value_counts()

plt.figure(figsize = (10,6))
g = sns.barplot(x=tech_sub.index[:2],y=tech_sub.values[:2])
g.set_xticklabels(g.get_xticklabels(),rotation=25)
g.set_title("Sub Category's by Ecommerce", fontsize=15)
g.set_xlabel("", fontsize=12)
g.set_ylabel("Count", fontsize=12)
plt.show()
plt.figure(figsize = (15,15))

stopwords = set(STOPWORDS)
wordcloud = WordCloud(
                          background_color='black',
                          stopwords=stopwords,
                          max_words=50,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(df_startups[df_startups['IndustryVertical'] == 'Ecommerce']['SubVertical']))

print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.title("WORD CLOUD - TITLES")
plt.axis('off')
plt.show()
df_startups.Date.replace((['12/05.2015', '13/04.2015','15/01.2015','22/01//2015']), \
                         ('12/05/2015','13/04/2015','15/01/2015','22/01/2015'), inplace=True)
df_startups['Date'] = pd.to_datetime(df_startups['Date'])

df_startups['Date_month_year'] = df_startups['Date'].dt.to_period("M")
df_startups['Date_year'] = df_startups['Date'].dt.to_period("A")
plt.figure(figsize=(14,10))
plt.subplot(211)
sns.countplot(x='Date_month_year', data=df_startups)
plt.xticks(rotation=90)
plt.xlabel('', fontsize=12)
plt.ylabel('Date Counting', fontsize=12)
plt.title("Count frequency Investiments Date ", fontsize=16)

plt.subplot(212)
sns.pointplot(x='Date_month_year', y='AmountInUSD_log', data=df_startups)
plt.xticks(rotation=90)
plt.xlabel('Dates', fontsize=12)
plt.ylabel('Amount Distribuition Log', fontsize=12)
plt.title("Money Distribuition by Month-Year", fontsize=16)

plt.subplots_adjust(wspace = 0.2, hspace = 0.4,top = 0.9)

plt.show()
Remarks = df_startups.Remarks.value_counts()

print("Remarks description")
print(Remarks[:10])

plt.figure(figsize=(10,5))

sns.barplot(x=Remarks.index[:10], y=Remarks.values[:10])
plt.xticks(rotation=45)
plt.xlabel('', fontsize=12)
plt.ylabel('Remarks Counting', fontsize=12)
plt.title("Count frequency Remarks ", fontsize=16)
plt.show()

plt.figure(figsize = (15,15))

stopwords = set(STOPWORDS)
wordcloud = WordCloud(
                          background_color='black',
                          stopwords=stopwords,
                          max_words=1500,
                          max_font_size=120, 
                          random_state=42
                         ).generate(str(df_startups['SubVertical']))

print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.title("WORD CLOUD - TITLES")
plt.axis('off')
plt.show()
plt.figure(figsize = (15,15))

stopwords = set(STOPWORDS)
wordcloud = WordCloud(
                          background_color='black',
                          stopwords=stopwords,
                          max_words=150,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(df_startups['InvestorsName']))

print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.title("WORD CLOUD - TITLES")
plt.axis('off')
plt.show()
plt.figure(figsize = (15,15))

stopwords = set(STOPWORDS)
wordcloud = WordCloud(
                          background_color='black',
                          stopwords=stopwords,
                          max_words=150,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(df_startups[df_startups['IndustryVertical'] == 'Technology']['InvestorsName']))

print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.title("WORD CLOUD - INVESTORS TECHNOLOGY")
plt.axis('off')
plt.show()
plt.figure(figsize = (15,15))

stopwords = set(STOPWORDS)
wordcloud = WordCloud(
                          background_color='black',
                          stopwords=stopwords,
                          max_words=150,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(df_startups[df_startups['IndustryVertical'] == 'Ecommerce']['InvestorsName']))

print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.title("WORD CLOUD - INVESTORS ECOMMERCE")
plt.axis('off')
plt.show()
