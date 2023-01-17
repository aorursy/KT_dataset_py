import pandas as pd

pd.options.display.max_columns = 100

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
stage_one = pd.read_csv('../input/gendered-pronoun-resolution/test_stage_1.tsv', sep='\t')

stage_two = pd.read_csv('../input/gendered-pronoun-resolution/test_stage_2.tsv', sep='\t')
display(stage_one.head())

display(stage_two.head())
print(stage_one.Text.iloc[0])

stng = stage_one.Text.iloc[0]

print(stng[191:204]) # The A-offset or where the Subject is

print(stng[207:211]) # The B-offset or where the object is

print(stng[274:278]) # The location of the pronoun
print(stage_two.Text.iloc[3])
stage_one.shape
stage_one['length'] = pd.Series([len(x) for x in stage_one.Text], index=stage_one.index)

stage_two['length'] = pd.Series([len(x) for x in stage_two.Text], index=stage_two.index)

print('Stage one data: ')







display(stage_one.describe())

display(stage_one.shape)

pronoun = stage_one.groupby("Pronoun")

print('\n\npronoun data: ')

display(pronoun.describe())

#display(pronoun.length)



plt.figure(figsize=(20,7))

sns.distplot(stage_one.length)

plt.xlabel("Frequency")

plt.ylabel("length of of text")





plt.figure(figsize=(12,7))

pronoun.size().sort_values().plot.bar()

plt.xticks(rotation=50)

plt.xlabel("Pronoun")

plt.ylabel("Frequency of pronoun")

plt.show()
print('Stage two data: ')

display(stage_two.describe())

display(stage_two.shape)

pronoun = stage_two.groupby("Pronoun")

print('pronoun data: ')

display(pronoun.describe())



plt.figure(figsize=(20,7))

sns.distplot(stage_two.length, color='g')

plt.xlabel("Frequency")

plt.ylabel("length of text")



plt.figure(figsize=(12,7))

pronoun.size().sort_values().plot.bar(color='g')

plt.xticks(rotation=50)

plt.xlabel("Pronoun")

plt.ylabel("Frequency of pronoun")

plt.show()


from os import path

from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
# Credit to DJ here, she had mentioned that pronouns can show as stop words (words that NLP tools will ignore as inconsequential)

# This cell collects the recognized pronouns and adds them to a list

non_stop = set(pronoun.Pronoun.unique().index)

print(non_stop)



non_stop = STOPWORDS - non_stop
#?WordCloud

#uncomment to see documentation for wordcloud, I had to reference this one alot
text = " ".join(sample for sample in stage_one.Text)

print("There are ", len(text)," words in the combination of all strings from 'Text' column.")





wordcloud = WordCloud(stopwords=STOPWORDS,width=1200,

    height=600).generate(text)



plt.figure(figsize=(20,15))

plt.imshow(wordcloud, interpolation='bilinear')

plt.title('WordCloud without pronouns', fontsize=26)

plt.axis("off")

plt.show()







wordcloud = WordCloud(stopwords=non_stop,width=1200,

    height=600).generate(text)



plt.figure(figsize=(20,15))

plt.imshow(wordcloud, interpolation='bilinear')

plt.title('WordCloud with pronouns', fontsize=26)

plt.axis("off")

plt.show()
text = " ".join(sample for sample in stage_two.Text)

print("There are ", len(text)," words in the combination of all strings from 'Text' column.")

print("this is 5x the number of words in the stage one dataset")

wordcloud = WordCloud(stopwords=STOPWORDS,width=1200,

    height=600).generate(text)



plt.figure(figsize=(20,15))

plt.imshow(wordcloud, interpolation='bilinear')

plt.title('WordCloud without pronouns', fontsize=26)

plt.axis("off")

plt.show()





wordcloud = WordCloud(stopwords=non_stop,width=1200,

    height=600).generate(text)



plt.figure(figsize=(20,15))

plt.imshow(wordcloud, interpolation='bilinear')

plt.title('WordCloud with pronouns', fontsize=26)

plt.axis("off")

plt.show()