import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import gc

from itertools import product

from xgboost import XGBClassifier

from xgboost import plot_importance

from datetime import datetime, date

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor



%matplotlib inline
# from google.colab import drive

# drive.mount('/content/drive')
# train = pd.read_csv('/content/drive/My Drive/kaggle/train.csv')

# test = pd.read_csv('/content/drive/My Drive/kaggle/test.csv')

# sample_submission = pd.read_csv('/content/drive/My Drive/kaggle/sample_submission.csv')
train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
train.head(5)
test.head(5)
sample_submission.head()
hist = train.hist(bins=10,column='target')

train = train.drop_duplicates().reset_index(drop=True)

hist1 = train.hist(bins=10,column='target')



print(hist,hist1)
plt.figure(figsize=(9,6))

sns.countplot(y=train.keyword, order = train.keyword.value_counts().iloc[:15].index)

plt.title('Top 15 keywords')

plt.show()
kw_d = train[train.target==1].keyword.value_counts().head(10)

kw_nd = train[train.target==0].keyword.value_counts().head(10)



plt.figure(figsize=(13,5))

plt.subplot(121)

sns.barplot(kw_d, kw_d.index, color='c')

plt.title('Top keywords for disaster tweets')

plt.subplot(122)

sns.barplot(kw_nd, kw_nd.index, color='y')

plt.title('Top keywords for non-disaster tweets')

plt.show()
# Most common locations

plt.figure(figsize=(9,6))

sns.countplot(y=train.location, order = train.location.value_counts().iloc[:15].index)

plt.title('Top 15 locations')

plt.show()
raw_loc = train.location.value_counts()

top_loc = list(raw_loc[raw_loc>=10].index)

top_only = train[train.location.isin(top_loc)]



top_l = top_only.groupby('location').mean()['target'].sort_values(ascending=False)

plt.figure(figsize=(14,6))

sns.barplot(x=top_l.index, y=top_l)

plt.axhline(np.mean(train.target))

plt.xticks(rotation=80)

plt.show()
import nltk

import re

nltk.download('punkt')
import string



def clean_text(text):

    text = re.sub(r'https?://\S+', '', text) # Remove link

    text = re.sub(r'\n',' ', text) # Remove line breaks

    text = re.sub('\s+', ' ', text).strip() # Remove leading, trailing, and extra spaces

    return text

train['text_clean'] = train['text'].apply(lambda x: clean_text(x))

test['text_clean'] = test['text'].apply(lambda x: clean_text(x))
train['text_len'] =  train.apply(lambda row: int(len(row['text'])), axis=1)

test['text_len'] =  test.apply(lambda row: int(len(row['text'])), axis=1)
train_len_max = max(train['text_len'])

test_len_max = max(test['text_len'])
train['text_len'] =  train.apply(lambda row: float(len(row['text'])/train_len_max), axis=1)

test['text_len'] =  test.apply(lambda row: float(len(row['text'])/test_len_max), axis=1)
train['text_len']
train['punctuation_count'] = train['text_clean'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))

test['punctuation_count'] = test['text_clean'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
train['hashtags']= train.apply(lambda row: re.findall(r"#(\w+)", row['text']), axis=1)

train['hashtags']= [i[0] if len(i)>0 else None for i in train['hashtags']]
test['hashtags']= test.apply(lambda row: re.findall(r"#(\w+)", row['text']), axis=1)

test['hashtags']= [i[0] if len(i)>0 else None for i in test['hashtags']]
train['count_mentions']= train.apply(lambda row: len(re.findall(r"@(\w+)", row['text'])), axis=1)

train['count_hashtags']= train.apply(lambda row: len(re.findall(r"#(\w+)", row['text'])), axis=1)

train['count_links']= train.apply(lambda row: len(re.findall(r"https?://\S+", row['text'])), axis=1)
test['count_mentions']= test.apply(lambda row: len(re.findall(r"@(\w+)", row['text'])), axis=1)

test['count_hashtags']= test.apply(lambda row: len(re.findall(r"#(\w+)", row['text'])), axis=1)

test['count_links']= test.apply(lambda row: len(re.findall(r"https?://\S+", row['text'])), axis=1)
def remove_punctuation(s):

  return s.translate(str.maketrans(' ', ' ', string.punctuation))
# train['text_clean'] = train.apply(lambda row: remove_punctuation(row['text_clean']), axis=1)
# test['text_clean'] = test.apply(lambda row: remove_punctuation(row['text_clean']), axis=1)
train['tokenized_sents'] = train.apply(lambda row: nltk.word_tokenize(row['text_clean']), axis=1)
test['tokenized_sents'] = test.apply(lambda row: nltk.word_tokenize(row['text_clean']), axis=1)
def countCL(new_text):

  return sum(map(str.isupper, new_text))
train["count_Capital_letters"] = train.apply(lambda row: int(countCL(row['text'])/20), axis=1)

test["count_Capital_letters"] = test.apply(lambda row: int(countCL(row['text'])/20), axis=1)
train["Capital_letters_precent"] = train.apply(lambda row: (countCL(row['text']))/len(row['text']), axis=1)

test["Capital_letters_precent"] = test.apply(lambda row: (countCL(row['text']))/len(row['text']), axis=1)
nltk.download('stopwords')

from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
from nltk.stem import PorterStemmer    

ps = PorterStemmer()
def removeStopWordsAndStem(sentence):

 l = [ps.stem(w) for w in sentence if not w in stop_words]

 return l
train['tokenized_sents'] = train.apply(lambda row: removeStopWordsAndStem(row['tokenized_sents']), axis=1)
test['tokenized_sents'] = test.apply(lambda row: removeStopWordsAndStem(row['tokenized_sents']), axis=1)
def isYear(sentence):

  count = 0

  for w in sentence:

    try:

      val = int(w)

      if val >1900 and val < 2020:

        count += 1

    except ValueError:

      pass

  return count
train["years_count"] = train.apply(lambda row: isYear(row['tokenized_sents']), axis=1)
test["years_count"] = test.apply(lambda row: isYear(row['tokenized_sents']), axis=1)
from sklearn.feature_extraction.text import TfidfVectorizer
list_tfidf = []

for i in range(train.shape[0]):

  list_tfidf = list_tfidf + train["tokenized_sents"][i]
vectorizer = TfidfVectorizer(min_df=0.0001,max_features=1000,ngram_range=(1,2))

p = vectorizer.fit_transform(list_tfidf).toarray()

print(vectorizer.get_feature_names())

print(len(vectorizer.get_feature_names()))
def createTfIdfArray(list_of_tokens,vectorizer_v):

  tmp = ' '.join(list_of_tokens)

  vector = vectorizer_v.transform([tmp])

  return vector.toarray()[0]
# list_tfidf_pos = []

# list_tfidf_neg = []

# for i in range(train.shape[0]):

#   if train['target'][i] == 1:

#     list_tfidf_pos = list_tfidf_pos + train["tokenized_sents"][i]

#   else:

#     list_tfidf_neg = list_tfidf_neg + train["tokenized_sents"][i]
# vectorizer_pos = TfidfVectorizer(max_features=500)

# vectorizer_pos.fit_transform(list_tfidf_pos).toarray()

# vectorizer_neg = TfidfVectorizer(max_features=500)

# vectorizer_neg.fit_transform(list_tfidf_neg).toarray()

# vectorizer_mix = TfidfVectorizer(min_df=0.00001,max_features=40)

# pos_and_neg = vectorizer_pos.get_feature_names()+vectorizer_neg.get_feature_names()

# vectorizer_mix.fit_transform(pos_and_neg).toarray()

# train_pos=train.query('target==1')

# train_neg=train.query('target==0')

# len(vectorizer_mix.get_feature_names())

# vectorizer = vectorizer_mix
train['tfIdf_vector'] = train.apply(lambda row: createTfIdfArray(row['tokenized_sents'],vectorizer), axis=1)
test['tfIdf_vector'] = test.apply(lambda row: createTfIdfArray(row['tokenized_sents'],vectorizer), axis=1)
# train_pos['tfIdf_vector'] = train_pos.apply(lambda row: createTfIdfArray(row['tokenized_sents'],vectorizer_pos), axis=1)

# train_neg['tfIdf_vector'] = train_neg.apply(lambda row: createTfIdfArray(row['tokenized_sents'],vectorizer_neg), axis=1)
map_keyword = {}

for count,i in enumerate(train.keyword.unique()):

  map_keyword[i] = count
def keyword_to_int(keyword):

  return map_keyword[keyword]
train["keyword"] = train.apply(lambda row: keyword_to_int(row['keyword']), axis=1)
test["keyword"] = test.apply(lambda row: keyword_to_int(row['keyword']), axis=1)
map_location = {}

for count,i in enumerate(train.location.unique()):

  map_location[i] = count
def location_to_int(location):

  try:

    return map_location[location]

  except:

    return -1
train["location"] = train.apply(lambda row: location_to_int(row['location']), axis=1)
test["location"] = test.apply(lambda row: location_to_int(row['location']), axis=1)
map_hashtags = {}

for count,i in enumerate(train.hashtags.unique()):

  map_hashtags[i] = count
def hashtags_to_int(hashtags):

  try:

    return map_hashtags[hashtags]

  except:

    return -1
train["hashtags"] = train.apply(lambda row: hashtags_to_int(row['hashtags']), axis=1)
test["hashtags"] = test.apply(lambda row: hashtags_to_int(row['hashtags']), axis=1)
new_train = train.copy()
new_test= test.copy()
def list_tfidf_by_index(count,df_new):

  l = []

  for k in df_new.tfIdf_vector:

    l.append(k[count])

  return l
for count in range(len(new_train.tfIdf_vector[0])):

    new_train["tfIdf_"+str(count)]= list_tfidf_by_index(count,new_train)
for count in range(len(new_test.tfIdf_vector[0])):

    new_test["tfIdf_"+str(count)]= list_tfidf_by_index(count,new_test)
new_train = new_train.drop(['id',"keyword","text_clean","location",'target',"text","tokenized_sents","tfIdf_vector","hashtags"],axis=1)

new_test = new_test.drop(['id',"keyword","text_clean","location","text","tokenized_sents","tfIdf_vector","hashtags"],axis=1)

new_train = new_train.fillna(0)

new_test = new_test.fillna(0)
new_test
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(new_train, train['target'], test_size=0.1, random_state=1)
# new_train = new_train.sample(frac=1)

# new_val = new_val.sample(frac=1)
#  X_train, X_val, y_train, y_val = new_train.drop(["target"],axis=1),new_val.drop(["target"],axis=1),new_train["target"],new_val["target"]
y_train.value_counts()
# Cross validation

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import ShuffleSplit



from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import MinMaxScaler

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics

clf = RandomForestClassifier(n_estimators=400,max_depth=1600, random_state=0)
clf.fit(X_train,y_train)

pred = clf.predict(X_val)

pscore = metrics.accuracy_score(y_val, pred)

pscore
# pred = pipeline.predict(new_test)

pred = clf.predict(new_test)
sample_submission['target'] = pred
sample_submission.to_csv('submission.csv',index = False)