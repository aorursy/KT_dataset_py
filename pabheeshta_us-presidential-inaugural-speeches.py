import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.style as style

import re

import string

import itertools

import collections

from bs4 import BeautifulSoup

from wordcloud import WordCloud

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize, RegexpTokenizer

style.use(['fivethirtyeight'])
speech_DF = pd.read_csv('../input/presidentialaddress/inaug_speeches.csv', encoding= 'latin1')
speech_DF.head()
speech_DF = speech_DF.drop(columns = 'Unnamed: 0')
speech_DF.head()
speech_DF.isnull().sum()
plt.figure(figsize = (7,7))

sns.countplot(data = speech_DF, x = 'Inaugural Address')

plt.xticks(rotation = 90)

plt.show()
speech_DF['Inaugural Address'].value_counts()
print("Total Number of Presidential Inaugural Addresses in US History: ", speech_DF.shape[0])
print("US Presidents Who Have Given Inaugural Speeches:\n\n", speech_DF['Name'].unique())
print("Total Number of US Presidents Who Have Given Inaugural Speeches: ", speech_DF['Name'].unique().size)
speech_DF.Name[speech_DF['Inaugural Address'] == 'Inaugural Address']
speech_DF.Name[speech_DF['Inaugural Address'] == 'Second Inaugural Address']
speech_DF.Name[speech_DF['Inaugural Address'] == 'Third Inaugural Address']
speech_DF_1 = speech_DF.copy()
speech_DF_1["Speech Length"] = speech_DF_1["text"].apply(lambda w : len(re.findall(r'\w+', w)))
speech_DF_1["Speech Year"] = pd.DatetimeIndex(speech_DF_1["Date"]).year
speech_DF_1.head()
plt.figure(figsize = (10,4))

sns.boxplot(data = speech_DF_1, x = "Speech Length", color = '#d43131')

plt.xlabel("Number of Words")

plt.show()
speech_DF_1['Speech Length'].describe()
plt.figure(figsize = (13, 7))

sns.lineplot(data = speech_DF_1, x = "Speech Year", y = "Speech Length")

plt.ylabel("Number of Words")

plt.ylim(0,9000)

plt.title("Length of Inaugural Addresses Over The Years")

plt.show()
print("Longest Speech:\n\n", speech_DF_1.iloc[speech_DF_1["Speech Length"].idxmax(axis=1), [0,1,2,4]])
print("Shortest Speech:\n\n", speech_DF_1.iloc[speech_DF_1["Speech Length"].idxmin(axis=1), [0,1,2,4]])
speech_DF_clean = speech_DF_1.copy()

speech_DF_clean = speech_DF_clean.drop(columns = "Speech Length")
for s in speech_DF_clean['text']:

     s = str(s)
def remove_u(s):

    no_u = re.sub(r'<.*?>', ' ', s)

    return no_u



speech_DF_clean['text'] = speech_DF_clean['text'].apply(lambda x: remove_u(x))
def remove_punc(s):

    no_punc = "".join([i for i in s if i not in string.punctuation])

    return no_punc



speech_DF_clean['text'] = speech_DF_clean['text'].apply(lambda x: remove_punc(x))
def remove_space(s):

    soup = BeautifulSoup(s, 'lxml')

    no_space = soup.get_text(strip = True)

    no_space = no_space.replace(u'\xa0', u'')

    return no_space



speech_DF_clean['text'] = speech_DF_clean['text'].apply(lambda x: remove_space(x))
tokenizer = RegexpTokenizer(r'\w+')

speech_DF_clean['text'] = speech_DF_clean['text'].apply(lambda x: tokenizer.tokenize(x.lower()))
def remove_stop_words(s):

    no_stop = [i for i in s if i not in stopwords.words('english')]

    return no_stop



speech_DF_clean['text'] = speech_DF_clean['text'].apply(lambda x: remove_stop_words(x))
def joining(s):

    joined_words = " ".join([i for i in s])

    return joined_words



speech_DF_clean['text'] = speech_DF_clean['text'].apply(lambda x: joining(x))
speech_DF_clean.text[1]
speeches = pd.Series(speech_DF_clean['text'].tolist()).astype(str)

plt.figure(figsize = (9, 9))

wcloud_all = WordCloud(width = 900, height = 900, colormap = 'magma', max_words = 150).generate(''.join(speeches))

plt.imshow(wcloud_all)

plt.tight_layout(pad = 0.2)

plt.axis('off')

plt.show()
speech_DF_token = speech_DF_clean.copy()
speech_DF_token['text'] = speech_DF_token['text'].apply(lambda x: tokenizer.tokenize(x.lower()))
speech_DF_token.head()
speech_list = list(itertools.chain.from_iterable(speech_DF_token['text']))
word_freq = collections.Counter(speech_list)
word_freq_DF = pd.DataFrame(word_freq.most_common(15), columns=['Words', 'Count'])



word_freq_DF.head(15)
plt.figure(figsize = (12, 7))

sns.barplot(data = word_freq_DF, x = "Words", y = "Count")

plt.ylabel("Frequency")

plt.ylim(0,600)

plt.xticks(rotation = 90)

plt.title("15 Most Frequent Words in Inaugural Speeches Overall")

plt.show()
speech_DF_token_1700 = speech_DF_token[speech_DF_token['Speech Year'] <= 1799]
print("Number of Inaugural Addresses in the 18th Century: ", speech_DF_token_1700.shape[0])

print("Number of Presidents in the 18th Century: ", speech_DF_token_1700['Name'].unique().size)
speech_list_1700 = list(itertools.chain.from_iterable(speech_DF_token_1700['text']))
word_freq_1700 = collections.Counter(speech_list_1700)

word_freq_DF_1700 = pd.DataFrame(word_freq_1700.most_common(10), columns=['Words', 'Count'])

word_freq_DF_1700.head(10)
plt.figure(figsize = (12, 7))

sns.barplot(data = word_freq_DF_1700, x = "Words", y = "Count")

plt.ylabel("Frequency")

plt.xticks(rotation = 90)

plt.title("10 Most Frequent Words in Inaugural Speeches in the 18th Century")

plt.show()
speech_DF_token_1800 = speech_DF_token[speech_DF_token['Speech Year'].between(1800, 1899, inclusive = True)]
print("Number of Inaugural Addresses in the 19th Century: ", speech_DF_token_1800.shape[0])

print("Number of Presidents in the 19th Century: ", speech_DF_token_1800['Name'].unique().size)
speech_list_1800 = list(itertools.chain.from_iterable(speech_DF_token_1800['text']))
word_freq_1800 = collections.Counter(speech_list_1800)

word_freq_DF_1800 = pd.DataFrame(word_freq_1800.most_common(10), columns=['Words', 'Count'])

word_freq_DF_1800.head(10)
plt.figure(figsize = (12, 7))

sns.barplot(data = word_freq_DF_1800, x = "Words", y = "Count")

plt.ylabel("Frequency")

plt.xticks(rotation = 90)

plt.title("10 Most Frequent Words in Inaugural Speeches in the 19th Century")

plt.show()
speech_DF_token_1900 = speech_DF_token[speech_DF_token['Speech Year'].between(1900, 1999, inclusive = True)]
print("Number of Inaugural Addresses in the 20th Century: ", speech_DF_token_1900.shape[0])

print("Number of Presidents in the 20th Century: ", speech_DF_token_1900['Name'].unique().size)
speech_list_1900 = list(itertools.chain.from_iterable(speech_DF_token_1900['text']))
word_freq_1900 = collections.Counter(speech_list_1900)

word_freq_DF_1900 = pd.DataFrame(word_freq_1900.most_common(10), columns=['Words', 'Count'])

word_freq_DF_1900.head(10)
plt.figure(figsize = (12, 7))

sns.barplot(data = word_freq_DF_1900, x = "Words", y = "Count")

plt.ylabel("Frequency")

plt.xticks(rotation = 90)

plt.title("10 Most Frequent Words in Inaugural Speeches in the 20th Century")

plt.show()
speech_DF_token_2000 = speech_DF_token[speech_DF_token['Speech Year'] >= 2000]
print("Number of Inaugural Addresses in the 21st Century: ", speech_DF_token_2000.shape[0])

print("Number of Presidents in the 21st Century: ", speech_DF_token_2000['Name'].unique().size)
speech_list_2000 = list(itertools.chain.from_iterable(speech_DF_token_2000['text']))
word_freq_2000 = collections.Counter(speech_list_2000)

word_freq_DF_2000 = pd.DataFrame(word_freq_2000.most_common(10), columns=['Words', 'Count'])

word_freq_DF_2000.head(10)
plt.figure(figsize = (12, 7))

sns.barplot(data = word_freq_DF_2000, x = "Words", y = "Count")

plt.ylabel("Frequency")

plt.xticks(rotation = 90)

plt.title("10 Most Frequent Words in Inaugural Speeches in the 21st Century")

plt.show()