import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



import ast # Abstract Syntax Trees; handling of JSON content



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# load data into data frame

df = pd.read_csv('../input/covid19-research-preprint-data/COVID-19-Preprint-Data_ver5.csv')
df.head()
df.describe(include='all')
tab_by_date = df['Date of Upload'].value_counts().sort_index()

plt.figure(figsize=(20,6))

tab_by_date.plot(kind='bar')

plt.grid()

plt.title('Date of Upload - Distribution')

plt.show()
df['Uploaded Site'].value_counts().plot(kind='bar')

plt.title('Uploaded Site')

plt.grid()

plt.show()
plt.figure(figsize=(8,6))

df['Number of Authors'].hist(bins=50)

plt.title('Number of Authors')

plt.show()
text = " ".join(title for title in df['Title of preprint'])

stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, max_font_size=50, max_words=500,

                      width = 600, height = 400,

                      background_color="white").generate(text)

plt.figure(figsize=(12,8))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
text = " ".join(abst for abst in df.Abstract)

stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, max_font_size=50, max_words=500,

                      width = 600, height = 400,

                      background_color="white").generate(text)

plt.figure(figsize=(12,8))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
# define keyword

my_keyword = 'Remdesivir'
def word_finder(i_word, i_text):

    found = str(i_text.lower()).find(i_word.lower())

    if found == -1:

        result = 0

    else:

        result = 1

    return result



# partial function for mapping

word_indicator_partial = lambda text: word_finder(my_keyword, text)

# build indicator vector (0/1) of hits

keyword_indicator = np.asarray(list(map(word_indicator_partial, df.Abstract)))

# number of hits

print('Number of hits for keyword <', my_keyword, '> : ', keyword_indicator.sum())
# add index vector as additional column

df['selection'] = keyword_indicator



# select only hits from data frame

df_hits = df[df['selection']==1]

# show results

df_hits
# look at an example: title...,

example_row = 1

df_hits['Title of preprint'].iloc[example_row]
# ... abstract

df_hits.Abstract.iloc[example_row]
# ... and authors

author_list = ast.literal_eval(df_hits.Authors.iloc[example_row])

author_list
# and corresponding institution counts

author_dict = ast.literal_eval(df_hits['Author(s) Institutions'].iloc[example_row])

author_dict
# finally a wordcloud of the selected results' abstracts

text = " ".join(abst for abst in df_hits.Abstract)

stopwords = set(STOPWORDS)

wordcloud = WordCloud(stopwords=stopwords, max_font_size=50, max_words=200,

                      width = 600, height = 400,

                      background_color="white").generate(text)

plt.figure(figsize=(12,8))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
# save results in CSV file for further processing

df_hits.to_csv('results.csv')
!pip install scispacy



# medium model

!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_md-0.2.4.tar.gz



# named entity extraction

# !pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_ner_bionlp13cg_md-0.2.4.tar.gz

!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_ner_bc5cdr_md-0.2.4.tar.gz    
import scispacy

import spacy



from spacy import displacy



import en_core_sci_md

import en_ner_bc5cdr_md
# look at an abstract

text = df_hits.Abstract.iloc[10]

text
nlp = en_core_sci_md.load()

doc = nlp(text)
# sentence parsing demo

displacy.render(next(doc.sents), style='dep', jupyter=True)
# Try basic entity extraction

doc.ents
# display entities

displacy.render(doc.sents, style='ent', jupyter=True)
# use specific Named Entity Recognition

nlp = en_ner_bc5cdr_md.load()
doc = nlp(text)
# Try more specific entity extraction

doc.ents
# display entities

displacy.render(doc.sents, style='ent', jupyter=True)