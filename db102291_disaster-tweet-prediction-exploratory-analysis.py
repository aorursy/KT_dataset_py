# Import packages

import re

import spacy

import numpy as np

import pandas as pd

import seaborn as sns

from matplotlib.pyplot import figure

import matplotlib.pyplot as plt

from collections import Counter

from spacy.lang.en.stop_words import STOP_WORDS

from wordcloud import WordCloud



# Custom settings

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)



cat_cols = ["#4f8c9d", "#94da40", "#7212ff", "#31d0a5", "#333a9e", "#b8b2f0", "#1c4c5e", "#8bd0eb", "#760796", "#39970e"]

sns.palplot(sns.color_palette(cat_cols))



%matplotlib inline

sns.set_style("whitegrid")

sns.set_palette(cat_cols)

sns.set_context("talk", font_scale=.9)



# Load data

nlp = spacy.load('en')

train = pd.read_csv('../input/nlp-getting-started/train.csv', index_col = 0)

test = pd.read_csv('../input/nlp-getting-started/test.csv', index_col = 0)
train.head()
display(train.isnull().sum().sort_values(ascending=False))
print('There are ' + str(len(train.keyword.unique())) + ' unique keywords.')
#Show the top 40 keywords for taget = 1

figure(figsize=(16, 16))



plt.subplot(1, 2, 1)

plt1 = sns.countplot(y="keyword", 

                     data=train,

                     hue=train.target,

                     order=train[train.target==1].keyword

                     .value_counts()

                     .iloc[:40].index)

plt1.set_ylabel('')

plt1.set_title('Top Keywords for Target = 1')



#Show the top 40 keywords for taget = 0

plt.subplot(1, 2, 2)

plt2 = sns.countplot(y="keyword", 

                     data=train,

                     hue=train.target,

                     order=train[train.target==0]

                     .keyword.value_counts()

                     .iloc[:40].index)

plt2.set_ylabel('')

plt2.set_title('Top Keywords for Target = 0')



plt.tight_layout(pad=3.0)
print('There are ' + str(len(train.location.unique())) + ' unique locations.')
#Show the top 10 locations for taget = 1

figure(figsize=(16, 6))



plt.subplot(1, 2, 1)

plt1 = sns.countplot(y="location", 

                     data=train,

                     hue=train.target,

                     order=train[train.target==1].location

                     .value_counts()

                     .iloc[:10].index)

plt1.set_ylabel('')

plt1.set_title('Top Locations for Target = 1')

plt1.legend(loc='lower right')



#Show the top 10 locations for taget = 0

plt.subplot(1, 2, 2)

plt2 = sns.countplot(y="location", 

                     data=train,

                     hue=train.target,

                     order=train[train.target==0].location

                     .value_counts()

                     .iloc[:10].index)

plt2.set_ylabel('')

plt2.set_title('Top Locations for Target = 0')

plt2.legend(loc='lower right')



plt.tight_layout(pad=3.0)
figure(figsize=(8, 6))

train["loc_bool"] = ["No Location" if pd.isnull(x) else "Location" for x in train["location"]]



plt1 = sns.countplot(x="loc_bool", 

                     data=train,

                     hue=train.target)

plt1.set_ylabel('')

plt1.set_xlabel('')

plt1.set_title('');
print('These are examples of disaster tweets: \n')

print(train[train.target==1]['text'][1:20])



print('\n')



print('These are examples of non-disaster tweets: \n')

print(train[train.target==0]['text'][1:20])
# Add boolean variable for URL

train["URL"] = train.text.str.match(r'[a-z]*[:.]+\S+', '')



# Remove URLs from tweets

train.text = [re.sub(r'[a-z]*[:.]+\S+', '', x) for x in train.text]

train.text = [re.sub(r'\&amp', '', x) for x in train.text]



# Create bar chart

figure(figsize=(8, 6))



df_p1 = train[["target", "URL"]].groupby(["target"]).mean().reset_index()

plt1 = sns.barplot(x="target", y="URL", data=df_p1)



vals = plt1.get_yticks()

plt1.set_yticklabels(['{:,.1%}'.format(x) for x in vals])

plt1.set_ylabel('')

plt1.set_xlabel('')

plt1.set_title('Percent of Tweets with URLs');
train["CAPS"] = [True if x == x.lower() else False for x in train.text]



figure(figsize=(8, 6))



df_p1 = train[["target", "CAPS"]].groupby(["target"]).mean().reset_index()



plt1 = sns.barplot(x="target",

                   y="CAPS",

                   data=df_p1)





vals = plt1.get_yticks()

plt1.set_yticklabels(['{:,.1%}'.format(x) for x in vals])

plt1.set_ylabel('')

plt1.set_xlabel('')

plt1.set_title('Percent of Capitalized Tweets');
train["Exclaim"] = [True if '!' in x else False for x in train.text]



figure(figsize=(8, 6))



df_p1 = train[["target", "Exclaim"]].groupby(["target"]).mean().reset_index()



plt1 = sns.barplot(x="target",

                   y="Exclaim",

                   data=df_p1)





vals = plt1.get_yticks()

plt1.set_yticklabels(['{:,.1%}'.format(x) for x in vals])

plt1.set_ylabel('')

plt1.set_xlabel('')

plt1.set_title('Percent of Exclamations');
#Setting up corpora for explorary analysis of text

text_TRUE = nlp(train[train['target']==1]['text'].str.cat(sep=' '))

text_FALSE = nlp(train[train['target']==0]['text'].str.cat(sep=' '))
# Remove stop words, punctuation, and spaces

all_TRUE = [token.text.lower() for token in text_TRUE

            if token.is_stop != True 

            and token.is_punct != True 

            and token.text != ' ' 

            and token.text != '  '

            and token.text != '\n' 

            and token.text != '\n\n']



all_FALSE = [token.text.lower() for token in text_FALSE 

             if token.is_stop != True 

             and token.is_punct != True 

             and token.text != ' ' 

             and token.text != '  '

             and token.text != '\n'

             and token.text != '\n\n']



# Create subsets that include only nouns or only verbs

nouns_TRUE = [token.text.lower() for token in text_TRUE if token.pos_ == "NOUN"]

nouns_FALSE = [token.text.lower() for token in text_FALSE if token.pos_ == "NOUN"]



verbs_TRUE = [token.text.lower() for token in text_TRUE if token.pos_ == "VERB"]

verbs_FALSE = [token.text.lower() for token in text_FALSE if token.pos_ == "VERB"]
# Find the most common words, nouns, and verbs

common_all_TRUE = pd.DataFrame(Counter(all_TRUE).most_common(20), columns = ["Word", "Frequency"])

common_all_FALSE = pd.DataFrame(Counter(all_FALSE).most_common(20), columns = ["Word", "Frequency"])

common_nouns_TRUE = pd.DataFrame(Counter(nouns_TRUE).most_common(20), columns = ["Word", "Frequency"])

common_nouns_FALSE = pd.DataFrame(Counter(nouns_FALSE).most_common(20), columns = ["Word", "Frequency"])

common_verbs_TRUE = pd.DataFrame(Counter(verbs_TRUE).most_common(20), columns = ["Word", "Frequency"])

common_verbs_FALSE = pd.DataFrame(Counter(verbs_FALSE).most_common(20), columns = ["Word", "Frequency"])
figure(figsize=(16, 6))



plt.subplot(1, 2, 1)

p1=sns.barplot(x=common_all_TRUE.Frequency, y=common_all_TRUE.Word);



plt.subplot(1, 2, 2)

p2=sns.barplot(x=common_all_FALSE.Frequency, y=common_all_FALSE.Word);



p1.set_title('Target = 1');

p2.set_title('Target = 0');

p2.set_ylabel('');
figure(figsize=(16, 6))



plt.subplot(1, 2, 1)

p1=sns.barplot(x=common_nouns_TRUE.Frequency, y=common_nouns_TRUE.Word);



plt.subplot(1, 2, 2)

p2=sns.barplot(x=common_nouns_FALSE.Frequency, y=common_nouns_FALSE.Word);



p1.set_title('Target = 1');

p2.set_title('Target = 0');

p2.set_ylabel('');
figure(figsize=(16, 6))



plt.subplot(1, 2, 1)

p1=sns.barplot(x=common_verbs_TRUE.Frequency, y=common_verbs_TRUE.Word);



plt.subplot(1, 2, 2)

p2=sns.barplot(x=common_verbs_FALSE.Frequency, y=common_verbs_FALSE.Word);



p1.set_title('Target = 1');

p2.set_title('Target = 0');

p2.set_ylabel('');