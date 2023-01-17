import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import scipy.stats

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

pd.options.display.max_columns = 1000
df = pd.read_csv("../input/train.csv")
df.head()
X = df['question_text']
y = df['target']
X.shape, y.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
X_train.shape, y_train.shape, X_test.shape, y_test.shape
df.info()
df['question_text'][df['question_text'] == ""].sum()
df['question_text'].shape, df['target'].shape
df['target'].unique()
sns.countplot(df['target'])
plt.xlabel('Predictions');
purcent_of_sincere = len(df['question_text'][df['target'] == 0]) / len(df['question_text']) * 100
purcent_of_unsincere = len(df['question_text'][df['target'] == 1]) / len(df['question_text']) * 100

sincere_len = len(df['question_text'][df['target'] == 0])
unsincere_len = len(df['question_text'][df['target'] == 1])

print("Purcent of sincere: {:.2f}%, {} questions".format(purcent_of_sincere, sincere_len))
print("Purcent of unsincere: {:.2f}%, {} questions".format(purcent_of_unsincere, unsincere_len))
sincere_lst_len = [len(df['question_text'][i]) for i in range(0, len(df['question_text'][df['target'] == 0])) if df['target'][i] == 0]
sincere_len_mean = np.array(sincere_lst_len).mean()
print("Mean of sincere questions: {:.0f} characters".format(sincere_len_mean))
unsincere_lst_len = [len(df['question_text'][i]) for i in range(0, len(df['question_text'][df['target'] == 1])) if df['target'][i] == 1]
unsincere_len_mean = np.array(unsincere_lst_len).mean()
print("Mean of unsincere questions: {:.0f} characters".format(unsincere_len_mean))
s1 = df[df['target'] == 0]['question_text'].str.len()
sns.distplot(s1, label='sincere')
s2 = df[df['target'] == 1]['question_text'].str.len()
sns.distplot(s2, label='unsincere')
plt.title('Lenght Distribution')
plt.legend();
first_word_unsincere = []
for sentence in df[df['target'] == 1]['question_text']:
    first_word_unsincere.append(sentence.split()[0])
from collections import Counter
counter_unsincere = Counter(first_word_unsincere)
counter_unsincere.most_common(10)
first_word_sincere = []
for sentence in df[df['target'] == 0]['question_text']:
    first_word_sincere.append(sentence.split()[0])
from collections import Counter
counter_sincere = Counter(first_word_sincere)
counter_sincere.most_common(10)
tokenized_docs = [word_tokenize(doc.lower()) for doc in X_train]
tokenized_docs[0]
alpha_tokens = [[t for t in doc if t.isalpha() == True] for doc in tokenized_docs]
alpha_tokens[0]
stop_words = stopwords.words('english')
no_stop_tokens = [[t for t in doc if t not in stop_words] for doc in alpha_tokens]
no_stop_tokens[0]
stemmer = PorterStemmer()
stemmed_tokens = [[stemmer.stem(t) for t in doc] for doc in no_stop_tokens]
stemmed_tokens[0]
X_temp = X_train.reset_index()
X_temp['temp'] = stemmed_tokens
X_temp.set_index('index', inplace=True)
X_temp.head()
X_temp = pd.concat([X_temp, y_train], axis=1, sort=False)
X_temp.head()
np_X_temp_index = np.array(X_temp.index)
lst = []
for idx in np_X_temp_index:
    lst.append(len(X_temp['temp'][idx]))
X_temp['count'] = lst
X_temp.head()
mean_count_sincere = X_temp['count'][X_temp['target'] == 0].mean()
print("Mean of preprocessed sincere words: {:.0f}".format(mean_count_sincere))
mean_count_unsincere = X_temp['count'][X_temp['target'] == 1].mean()
print("Mean of preprocessed unsincere words: {:.0f}".format(mean_count_unsincere))
X_train_clean = [" ".join(x_t) for x_t in stemmed_tokens]
X_train_clean
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
vectorizer = TfidfVectorizer(stop_words='english')
svd = TruncatedSVD(random_state=42)
preprocessing_pipe = Pipeline([('vectorizer', vectorizer), ('svd', svd)])
lsa_train = preprocessing_pipe.fit_transform(X_train_clean)
lsa_train.shape
sns.scatterplot(x=lsa_train[:10000, 0], y=lsa_train[:10000, 1], hue=y_train[:10000]);
components = pd.DataFrame(data=svd.components_, columns=preprocessing_pipe.named_steps['vectorizer'].get_feature_names(), index=['component_0', 'component_1'])
components
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
for i, ax in enumerate(axes.flat):
    components.iloc[i].sort_values(ascending=False)[:10].sort_values().plot.barh(ax=ax)
def cleaning(df):
    tokenized_docs = [word_tokenize(doc.lower()) for doc in df]
    alpha_tokens = [[t for t in doc if t.isalpha() == True] for doc in tokenized_docs]
    no_stop_tokens = [[t for t in doc if t not in stop_words] for doc in alpha_tokens]
    stemmed_tokens = [[stemmer.stem(t) for t in doc] for doc in no_stop_tokens]
    df_clean = [" ".join(x_t) for x_t in stemmed_tokens]
    return df_clean
X_test_clean = cleaning(X_test)
X_test_clean
cvec_unigram = CountVectorizer(stop_words='english').fit(X_train_clean)
mb = MultinomialNB()
pipe = make_pipeline(cvec_unigram, mb)
pipe.fit(X_train_clean, y_train)
pipe.score(X_train_clean, y_train)
pipe.score(X_test_clean, y_test)
y_pred = pipe.predict(X_test_clean)
confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
scores = cross_val_score(pipe, X_train_clean, y_train, cv=5, scoring='f1')
scores
print("mean: {}".format(scores.mean()))
print("std: {}".format(scores.std()))
cvec_bigram = CountVectorizer(stop_words='english', ngram_range=(2, 2)).fit(X_train_clean)
mb = MultinomialNB()
pipe_bi = make_pipeline(cvec_bigram, mb)
pipe_bi.fit(X_train_clean, y_train)
pipe_bi.score(X_train_clean, y_train)
pipe_bi.score(X_test_clean, y_test)
y_pred_bi = pipe_bi.predict(X_test_clean)
confusion_matrix(y_test, y_pred_bi)
print(classification_report(y_test, y_pred_bi))
scores_bi = cross_val_score(pipe_bi, X_train_clean, y_train, cv=5, scoring='f1')
scores_bi
print("mean: {}".format(scores_bi.mean()))
print("std: {}".format(scores_bi.std()))
cvec_trigram = CountVectorizer(stop_words='english', ngram_range=(3, 3)).fit(X_train_clean)
mb = MultinomialNB()
pipe_tri = make_pipeline(cvec_trigram, mb)
pipe_tri.fit(X_train_clean, y_train)
pipe_tri.score(X_train_clean, y_train)
pipe_tri.score(X_test_clean, y_test)
y_pred_tri = pipe_tri.predict(X_test_clean)
confusion_matrix(y_test, y_pred_tri)
print(classification_report(y_test, y_pred_tri))
scores_tri = cross_val_score(pipe_tri, X_train_clean, y_train, cv=5, scoring='f1')
scores_tri
print("mean: {}".format(scores_tri.mean()))
print("std: {}".format(scores_tri.std()))