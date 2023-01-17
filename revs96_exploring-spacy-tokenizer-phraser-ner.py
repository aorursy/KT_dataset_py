# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
%matplotlib inline

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
nlp = spacy.load('en_core_web_lg')

review_data = pd.read_csv('/kaggle/input/medium-articles/articles.csv')
review_data.head()
review_data.size
data = pd.read_json('/kaggle/input/medium-articles-as-json/articles.json')
data.head()
menu = ["machine learning", "ML"]
from spacy.matcher import PhraseMatcher

index_of_review_to_test_on = 3
text_to_test_on = data.text.iloc[index_of_review_to_test_on]

##print(text_to_test_on)

# Load the SpaCy model
nlp = spacy.blank('en')

# Create the tokenized version of text_to_test_on
review_doc = nlp(text_to_test_on)

# Create the PhraseMatcher object. The tokenizer is the first argument. Use attr = 'LOWER' to make consistent capitalization
matcher = PhraseMatcher(nlp.vocab, attr='LOWER')

# Create a list of tokens for each item in the menu
menu_tokens_list = [nlp(item) for item in menu]

#print(menu_tokens_list)

# Add the item patterns to the matcher. 
# Look at https://spacy.io/api/phrasematcher#add in the docs for help with this step
# Then uncomment the lines below 

# 
matcher.add("MENU",            # Just a name for the set of rules we're matching to
            None,              # Special actions to take on matched words
            *menu_tokens_list  
            )

# Find matches in the review_doc
matches = matcher(review_doc)
print(matches)

for match in matches:
    print(f"Token number {match[2]}: {review_doc[match[1]:match[2]]}")
nlp = spacy.load('en')
doc = nlp(data.text.iloc[4])
print(f"Token \t\tLemma \t\tStopword".format('Token', 'Lemma', 'Stopword'))
print("-"*40)
for token in doc[250:300]:
    print(f"{str(token)}\t\t{token.lemma_}\t\t{token.is_stop}")
nlp = spacy.load('en_core_web_lg')
articles = pd.read_csv("/kaggle/input/medium-articles/articles.csv")
def explain_text_entities(text):
    doc = nlp(text)
    for ent in doc.ents:
        print(f'Entity: {ent}, Label: {ent.label_}, {spacy.explain(ent.label_)}')
explain_text_entities(articles['text'][9])
one_sentence = articles['text'][0]
doc = nlp(one_sentence)
spacy.displacy.render(doc, style='ent',jupyter=True)
def redact_names(text):
    doc = nlp(text)
    redacted_sentence = []
    for ent in doc.ents:
        ent.merge()
    for token in doc:
        if token.ent_type_ == "PERSON":
            redacted_sentence.append("[REDACTED_PERSON]")
        else:
            redacted_sentence.append(token.string)
    return "".join(redacted_sentence)
print("**Before**")
one_sentence = articles['text'][0]
doc = nlp(one_sentence)
spacy.displacy.render(doc, style='ent',jupyter=True)
print("**After**")
one_sentence = redact_names(articles['text'][0])
doc = nlp(one_sentence)
spacy.displacy.render(doc, style='ent',jupyter=True)

example_text = articles['text'][9]
doc = nlp(example_text)
spacy.displacy.render(doc, style='ent', jupyter=True)

for idx, sentence in enumerate(doc.sents):
    for noun in sentence.noun_chunks:
        print(f"sentence {idx+1} has noun chunk '{noun}'")
one_sentence = articles['text'][300]
doc = nlp(one_sentence)
spacy.displacy.render(doc, style='ent', jupyter=True)

for token in doc:
    print(token, token.pos_)
text = articles['text'].str.cat(sep=' ')
# spaCy enforces a max limit of 1000000 characters for NER and similar use cases.
# Since `text` might be longer than that, we will slice it off here
max_length = 1000000-1
text = text[:max_length]

# removing URLs and '&amp' substrings using regex
import re
url_reg  = r'[a-z]*[:.]+\S+'
text   = re.sub(url_reg, '', text)
noise_reg = r'\&amp'
text   = re.sub(noise_reg, '', text)
doc = nlp(text)
items_of_interest = list(doc.noun_chunks)
# each element in this list is spaCy's inbuilt `Span`, which is not useful for us
items_of_interest = [str(x) for x in items_of_interest]
# so we've converted it to string
import seaborn as sns
df_nouns = pd.DataFrame(items_of_interest, columns=["What is it about"])
plt.figure(figsize=(7,8))
sns.countplot(y="What is it about",
             data=df_nouns,
             order=df_nouns["What is it about"].value_counts().iloc[:10].index)
plt.show()

trump_topics = []
for token in doc:
    if (not token.is_stop) and (token.pos_ == "NOUN") and (len(str(token))>2):
        trump_topics.append(token)
        
trump_topics = [str(x) for x in trump_topics]
df_nouns = pd.DataFrame(trump_topics, columns=["Article deals with"])
df_nouns
plt.figure(figsize=(7,8))
sns.countplot(y="Article deals with",
             data=df_nouns,
             order=df_nouns["Article deals with"].value_counts().iloc[:10].index)
plt.show()
trump_topics = []
for ent in doc.ents:
    if ent.label_ not in ["PERCENT", "CARDINAL", "DATE"]:
#         print(ent.text,ent.label_)
        trump_topics.append(ent.text.strip())
df_ttopics = pd.DataFrame(trump_topics, columns=["Deals with"])
plt.figure(figsize=(7,8))
sns.countplot(y="Deals with",
             data=df_ttopics,
             order=df_ttopics["Deals with"].value_counts().iloc[1:11].index)
plt.show()
# from collections import Counter
# item_counter = Counter(items_of_interest)
# item_counter.most_common()
from spacy.lang.en.stop_words import STOP_WORDS
from wordcloud import WordCloud
plt.figure(figsize=(10,5))
wordcloud = WordCloud(background_color="white",
                      stopwords = STOP_WORDS,
                      max_words=45,
                      max_font_size=30,
                      random_state=42
                     ).generate(str(trump_topics))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
from spacy.matcher import Matcher
doc = nlp(text)
matcher = Matcher(nlp.vocab)
matched_sents = [] # collect data of matched sentences to be visualized

def collect_sents(matcher, doc, i, matches, label='MATCH'):
    """
    Function to help reformat data for displacy visualization
    """
    match_id, start, end = matches[i]
    span = doc[start : end]  # matched span
    sent = span.sent  # sentence containing matched span
    
    # append mock entity for match in displaCy style to matched_sents
    
    if doc.vocab.strings[match_id] == 'TENSORFLOW':  # don't forget to get string!
        match_ents = [{'start': span.start_char - sent.start_char,
                   'end': span.end_char - sent.start_char,
                   'label': 'TENSORFLOW'}]
        matched_sents.append({'text': sent.text, 'ents': match_ents })
    elif doc.vocab.strings[match_id] == 'MACHINE LEARNING':  # don't forget to get string!
        match_ents = [{'start': span.start_char - sent.start_char,
               'end': span.end_char - sent.start_char,
               'label': 'MACHINE LEARNING'}]
        matched_sents.append({'text': sent.text, 'ents': match_ents })
    elif doc.vocab.strings[match_id] == 'WE':  # don't forget to get string!
        match_ents = [{'start': span.start_char - sent.start_char,
               'end': span.end_char - sent.start_char,
               'label': 'WE'}]
        matched_sents.append({'text': sent.text, 'ents': match_ents })
    
# declare different patterns
russia_pattern = [{'LOWER': 'MACHINE LEARNING'}, {'LEMMA': 'be'}, {'POS': 'ADV', 'OP': '*'},
           {'POS': 'ADJ'}]
democrats_pattern = [{'LOWER': 'TENSORFLOW'}, {'LEMMA': 'be'}, {'POS': 'ADV', 'OP': '*'},
           {'POS': 'ADJ'}]
i_pattern = [{'LOWER': 'WE'}, {'LEMMA': 'be'}, {'POS': 'ADV', 'OP': '*'},
           {'POS': 'ADJ'}]

matcher.add('TENSORFLOW', collect_sents, democrats_pattern)  # add pattern
matcher.add('MACHINE LEARNING', collect_sents, russia_pattern)  # add pattern
matcher.add('WE', collect_sents, i_pattern)  # add pattern
matches = matcher(doc)

spacy.displacy.render(matched_sents, style='ent', manual=True, jupyter=True,  options = {'colors': {'WE': '#6290c8', 'MACHINE LEARNING': '#cc2936', 'TENSORFLOW':'#f2cd5d'}})

print(matched_sents[:3])
