import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns



from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



import os
df = pd.read_csv('../input/data-engineer-jobs/DataEngineer.csv')

df.head()
df.shape
df.columns
df['Job Title'].value_counts()
# show top 25 only

temp = df['Job Title'].value_counts()

sns.barplot(x=temp.index[0:25], y=temp[0:25])

plt.title('Top 25 - Job Title')

plt.xticks(rotation=90)

plt.grid()

plt.show()
df['Salary Estimate'].value_counts()
# show top 25 only

temp = df['Salary Estimate'].value_counts()

sns.barplot(x=temp.index[0:25], y=temp[0:25])

plt.title('Top 25 - Salary Estimate')

plt.xticks(rotation=90)

plt.grid()

plt.show()
def aux1(i_string):

    return pd.to_numeric((i_string.split('K')[0]).split('$')[1])



def aux2(i_string):

    return pd.to_numeric((i_string.split('K')[1]).split('$')[1])
df['Salary_LoB'] = list(map(aux1, df['Salary Estimate']))

df['Salary_UpB'] = list(map(aux2, df['Salary Estimate']))
df['Salary_Mid'] = (df['Salary_LoB'] + df['Salary_UpB'])/2 
df.Salary_LoB.hist(bins=25)

plt.title('Salary Lower Bound (in 1000 USD)')

plt.show()
df.Salary_LoB.describe()
df.Salary_UpB.hist(bins=25)

plt.title('Salary Upper Bound (in 1000 USD)')

plt.show()
df.Salary_UpB.describe()
df.Salary_Mid.hist(bins=25)

plt.title('Salary Mid Point of range (in 1000 USD)')

plt.show()
df.Salary_Mid.describe()
df.Rating.plot(kind='hist')

plt.title('Rating')

plt.grid()

plt.show()
df['Company Name'].value_counts()
# utility function for text cleaning

def chop_name(i_string):

    return i_string.split('\n')[0]
# show top 25 only

temp = df['Company Name'].value_counts()

sns.barplot(x=list(map(chop_name,temp.index[0:25])), y=temp[0:25])

plt.title('Top 25 - Company Name')

plt.xticks(rotation=90)

plt.grid()

plt.show()
# add clean company name as addition column

df['Company'] = list(map(chop_name,df['Company Name']))
df['Headquarters'].value_counts()
# show top 25 only

temp = df['Headquarters'].value_counts()

sns.barplot(x=temp.index[0:25], y=temp[0:25])

plt.title('Top 25 - Headquarters')

plt.xticks(rotation=90)

plt.grid()

plt.show()
# Size

df['Size'] = df['Size'].replace("-1","Unknown") # merge "-1" into "Unknown"

df['Size'].value_counts().plot(kind='bar')

plt.grid()

plt.show()
df.Founded.plot(kind='hist')

plt.title('Founded')

plt.grid()

plt.show()
# show Founded w/o missings (-1)

temp = df.Founded[df.Founded>-1]

plt.hist(temp,50)

plt.title('Founded, excluding missing values')

plt.grid()

plt.show()
# Founded summary

temp.describe()
# Type of ownership

df['Type of ownership'] = df['Type of ownership'].replace("-1","Unknown") # merge "-1" into "Unknown"

df['Type of ownership'].value_counts().plot(kind='bar')

plt.grid()

plt.show()
df['Industry'].value_counts()
# show top 25 only

temp = df['Industry'].value_counts()

sns.barplot(x=temp.index[0:25], y=temp[0:25])

plt.title('Top 25 - Industry')

plt.xticks(rotation=90)

plt.grid()

plt.show()
df['Sector'].value_counts().plot(kind='bar')

plt.title('Sector')

plt.grid()

plt.show()
# Revenue

df['Revenue'] = df['Revenue'].replace("-1","Unknown / Non-Applicable") # merge "-1" into "Unknown..."

df['Revenue'].value_counts().plot(kind='bar')

plt.title('Revenue')

plt.grid()

plt.show()
df['Easy Apply'].value_counts().plot(kind='bar')

plt.title('Easy Apply')

plt.grid()

plt.show()
# means by company

df_means = df.groupby('Company').mean()

df_means.head()
sel_company = 'Amazon'

df_means[df_means.index==sel_company]
df_temp = df[df.Company==sel_company]

df_temp.Salary_Mid.hist()

plt.title(sel_company)

plt.show()
sel_company = 'Apple'

df_means[df_means.index==sel_company]
df_temp = df[df.Company==sel_company]

df_temp.Salary_Mid.hist()

plt.title(sel_company)

plt.show()
sel_company = 'Lockheed Martin'

df_means[df_means.index==sel_company]
df_temp = df[df.Company==sel_company]

df_temp.Salary_Mid.hist()

plt.title(sel_company)

plt.show()
sel_company = 'Google'

df_means[df_means.index==sel_company]
df_temp = df[df.Company==sel_company]

df_temp.Salary_Mid.hist()

plt.title(sel_company)

plt.show()
stopwords = set(STOPWORDS)

text = " ".join(txt for txt in df['Job Description'])
wordcloud = WordCloud(stopwords=stopwords, max_font_size=50, max_words=500,

                      width = 600, height = 400,

                      background_color="white").generate(text)

plt.figure(figsize=(12,8))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
sel_company = 'Apple'

df_temp = df[df.Company==sel_company]

text = " ".join(txt for txt in df_temp['Job Description'])



wordcloud = WordCloud(stopwords=stopwords, max_font_size=50, max_words=500,

                      width = 600, height = 400,

                      background_color="white").generate(text)

plt.figure(figsize=(12,8))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
sel_company = 'Amazon'

df_temp = df[df.Company==sel_company]

text = " ".join(txt for txt in df_temp['Job Description'])



wordcloud = WordCloud(stopwords=stopwords, max_font_size=50, max_words=500,

                      width = 600, height = 400,

                      background_color="white").generate(text)

plt.figure(figsize=(12,8))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
sel_company = 'Google'

df_temp = df[df.Company==sel_company]

text = " ".join(txt for txt in df_temp['Job Description'])



wordcloud = WordCloud(stopwords=stopwords, max_font_size=50, max_words=500,

                      width = 600, height = 400,

                      background_color="white").generate(text)

plt.figure(figsize=(12,8))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
sel_sector = 'Finance'

df_temp = df[df.Sector==sel_sector]

text = " ".join(txt for txt in df_temp['Job Description'])



wordcloud = WordCloud(stopwords=stopwords, max_font_size=50, max_words=500,

                      width = 600, height = 400,

                      background_color="white").generate(text)

plt.figure(figsize=(12,8))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
sel_sector = 'Manufacturing'

df_temp = df[df.Sector==sel_sector]

text = " ".join(txt for txt in df_temp['Job Description'])



wordcloud = WordCloud(stopwords=stopwords, max_font_size=50, max_words=500,

                      width = 600, height = 400,

                      background_color="white").generate(text)

plt.figure(figsize=(12,8))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()