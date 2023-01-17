# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
import re
from sklearn.model_selection import train_test_split
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.metrics import classification_report

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import glob
path = '/kaggle/input/coronavirus-covid19-tweets-late-april/'
all_files = glob.glob(os.path.join(path, "*.CSV"))

df_from_each_file = (pd.read_csv(f) for f in all_files)
concatenated_df   = pd.concat(df_from_each_file, ignore_index=True)
df = concatenated_df[['text', 'lang']]
filter = df.mask(lambda x: x['lang'] != 'en').dropna()
filter.head()
texts = filter['text'].drop_duplicates()
joy_icons = '😁|😂|😃|😄|😅|😆|😉|😊|😍'

joy = pd.Series(texts, dtype="string", name="joy").str.contains(joy_icons)
joy
sad_icons = '😓|😖|😢|😭|😰|😱|🙍|🙎'

sad = pd.Series(texts, dtype="string", name="sad").str.contains(sad_icons)
sad
angry_icons = '😠|😡|😤|🤬|👿|💀|☠'

angry = pd.Series(texts, dtype="string", name="angry").str.contains(angry_icons)
angry
hope_icons = '😷|🙋|🙌|🙏'

hope = pd.Series(texts, dtype="string", name="hope").str.contains(hope_icons)
hope
df = pd.concat([texts, joy, sad, angry, hope], axis=1)
df
texts_classified = df[(df['joy'] | df['sad'] | df['angry'] | df['hope'])]
texts_not_classified = df[~(df['joy'] | df['sad'] | df['angry'] | df['hope'])]['text']
count = texts_classified.apply(lambda row: row['joy'] + row['sad'] + row['angry'] + row['hope'], axis=1)
count = count.rename("count")
texts_classified = pd.concat([texts_classified, count], axis=1)
texts_classified = texts_classified[texts_classified['count'] == 1]
texts_classified
def label_feeling (row):
   if row['joy'] :
      return 'JOY'
   if row['sad'] :
      return 'SAD'
   if row['angry'] :
      return 'ANGRY'
   if row['hope'] :
      return 'HOPE'
   return 'NONE'

texts_classified['target'] = texts_classified.apply(lambda row: label_feeling(row), axis=1)
texts_classified
texts_classified = texts_classified[['text', 'target']]
texts_classified.head()
def  clean_text(df, text_field):
    df[text_field] = df[text_field].str.lower()
    df[text_field] = df[text_field].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))  
    
    # remove numbers
    df[text_field] = df[text_field].apply(lambda elem: re.sub(r"\d+", "", elem))
    
    return df
clean_text(texts_classified, 'text')
texts_classified.head()
stop = stopwords.words('english')

texts_classified['text'] = texts_classified['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
texts_classified
texts_classified['target'].value_counts()
hope_target = texts_classified[texts_classified['target'] == 'HOPE'].sample(8914)
joy_target = texts_classified[texts_classified['target'] == 'JOY'].sample(8914)
sad_target = texts_classified[texts_classified['target'] == 'SAD'].sample(8914)
angry_target = texts_classified[texts_classified['target'] == 'ANGRY'].sample(8914)
balanced_data = pd.concat([hope_target, joy_target, sad_target, angry_target], ignore_index=True)
balanced_data
X_train, X_test, y_train, y_test = train_test_split(balanced_data['text'], balanced_data['target'], test_size=0.33, random_state=0, stratify=balanced_data['target'])
text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
text_clf = text_clf.fit(X_train, y_train)
predicted = text_clf.predict(X_test)
print(classification_report(y_test, predicted))
pd.Series(predicted).value_counts()
clean_data = texts_not_classified.str.lower()
clean_data = clean_data.apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))  
clean_data = clean_data.apply(lambda elem: re.sub(r"\d+", "", elem))
clean_data
clean_data = clean_data.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
clean_data
results = text_clf.predict(clean_data)
pd.Series(results).value_counts()
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk import sent_tokenize, word_tokenize, pos_tag
 
 
lemmatizer = WordNetLemmatizer()
 
 
def penn_to_wn(tag):
    """
    Convert between the PennTreebank tags to simple Wordnet tags
    """
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None
 
 
def clean_text(text):
    text = text.replace("<br />", " ")
 
    return text
 
 
def swn_polarity(text):
    """
    Return a sentiment polarity: 0 = negative, 1 = positive
    """
 
    sentiment = 0.0
    tokens_count = 0
 
    text = clean_text(text)
 
 
    raw_sentences = sent_tokenize(text)
    for raw_sentence in raw_sentences:
        tagged_sentence = pos_tag(word_tokenize(raw_sentence))
 
        for word, tag in tagged_sentence:
            wn_tag = penn_to_wn(tag)
            if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
                continue
 
            lemma = lemmatizer.lemmatize(word, pos=wn_tag)
            if not lemma:
                continue
 
            synsets = wn.synsets(lemma, pos=wn_tag)
            if not synsets:
                continue
 
            # Take the first sense, the most common
            synset = synsets[0]
            swn_synset = swn.senti_synset(synset.name())
 
            sentiment += swn_synset.pos_score() - swn_synset.neg_score()
            tokens_count += 1
 
    # judgment call ? Default to positive or negative
    if not tokens_count:
        return 0
 
    # sum greater than 0 => positive sentiment
    if sentiment >= 0:
        return 1
 
    # negative sentiment
    return 0
pred_y = [swn_polarity(text) for text in clean_data]
pd.Series(pred_y).value_counts()