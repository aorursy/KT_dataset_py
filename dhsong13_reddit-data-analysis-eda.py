# Image

from PIL import Image



# Python Collectino

from collections import Counter



# FOR Loop Verbose

from tqdm import tqdm



# System

import os



# String

import string



# Natural Language Processing

import nltk

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from nltk.tag import pos_tag

from nltk.stem import WordNetLemmatizer



# Date and Time

import datetime



# Dataframe

import pandas as pd



# Visualization

import matplotlib.pyplot as plt

import seaborn as sns

from wordcloud import WordCloud



# Numerical Data

import numpy as np



# Machine Learning

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_validate

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.manifold import TSNE

from sklearn.decomposition import TruncatedSVD

from sklearn.decomposition import LatentDirichletAllocation

from sklearn.ensemble import GradientBoostingClassifier
data_path = '/usr/share/nltk_data'

print(data_path)

if not os.path.exists(data_path):

    nltk.download()

nltk.data.path.append(data_path)
pd.options.display.max_rows = 499

pd.options.display.max_columns = 499

pd.options.mode.chained_assignment = None
%matplotlib inline
raw = pd.read_csv('/kaggle/input/dataisbeautiful/r_dataisbeautiful_posts.csv', encoding='utf-8')

raw
raw.info()
raw.describe(include='all')
raw.isna().sum()
cleaned = raw.copy()
cleaned.title = cleaned.title.fillna('null')
cleaned[cleaned.title == 'null']
columns = ['author_flair_text', 'removed_by', 'total_awards_received', 'awarders']

cleaned = cleaned.drop(columns, axis=1)

cleaned
cleaned.isna().sum()
def utc_to_datetime(data):

    data['year'] = data['created_utc'].apply(lambda utc: datetime.datetime.fromtimestamp(utc).year)

    data['month'] = data['created_utc'].apply(lambda utc: datetime.datetime.fromtimestamp(utc).month)

    data['day'] = data['created_utc'].apply(lambda utc: datetime.datetime.fromtimestamp(utc).day)    

    data['hour'] = data['created_utc'].apply(lambda utc: datetime.datetime.fromtimestamp(utc).hour)

    data['minute'] = data['created_utc'].apply(lambda utc: datetime.datetime.fromtimestamp(utc).minute)

    data['second'] = data['created_utc'].apply(lambda utc: datetime.datetime.fromtimestamp(utc).second)    
utc_to_datetime(cleaned)

cleaned
cleaned['original_content'] = cleaned['title'].str.contains('[OC]').astype(int)

cleaned
def get_wordnet_tag(tag):

    if tag == 'ADJ':

        return 'j'

    elif tag == 'VERB':

        return 'v'

    elif tag == 'NOUN':

        return 'n'

    elif tag == 'ADV':

        return 'r'

    else:

        return 'n'
def lemmatize_text(title):

    stop = stopwords.words('english')

    lemmatizer = WordNetLemmatizer()



    words = list()

    title = word_tokenize(title)

    for word, tag in pos_tag(title):

        tag = get_wordnet_tag(tag)

        word = lemmatizer.lemmatize(word, tag)

        if word not in stop:

            words.append(word)

    

    return ' '.join(words)        
def clean_text(dataset):

    

    tqdm.pandas()

    

    dataset['title_cleaned'] = dataset['title'].str.lower()

    dataset['title_cleaned'] = dataset['title_cleaned'].str.replace(r'\[oc\]', ' ')

    pattern_link = r'https?://[^\s]+|www\.[^\s]+|[^\s]+\.com[^\s]*|[^\s]+\.org[^\s]*|[^\s]+\.html[^\s]*'

    dataset['title_cleaned'] = dataset['title_cleaned'].str.replace(pattern_link, ' link ')

    

    pattern_punctuation = r'[' + string.punctuation + '’]'

    dataset['title_cleaned'] = dataset['title_cleaned'].str.replace(pattern_punctuation, '')

    dataset['title_cleaned'] = dataset['title_cleaned'].str.replace(r' [\d]+ |^[\d]+ | [\d]+$', ' ')

    dataset['title_cleaned'] = dataset['title_cleaned'].str.replace(r'[^\w\d\s]+', ' ')

    dataset['title_cleaned'] = dataset['title_cleaned'].progress_apply(lambda title: lemmatize_text(title))

    

    dataset['title_cleaned'] = dataset['title_cleaned'].str.replace(r'\s[\s]+', ' ')
clean_text(cleaned)

cleaned
cleaned['over_18'] = cleaned['over_18'].apply(lambda x: int(x))

cleaned
def boxplot(data, feature, base):

    assert base in ['over_18', 'original_content']

    

    plt.figure(figsize=(30, 12))

    sns.boxplot(x=base, y=feature, data=data)

    plt.show()
cleaned['score'].value_counts()
boxplot(cleaned, 'score', 'over_18')
boxplot(cleaned, 'score', 'original_content')
cleaned['num_comments'].value_counts()
boxplot(cleaned, 'num_comments', 'over_18')
boxplot(cleaned, 'num_comments', 'original_content')
def countplot(data, by='year'):

    assert by in ['year', 'month', 'day']

    data_copy = data.copy()

    data_copy['year'] = data_copy['year'].astype(str)

    data_copy['month'] = data_copy['month'].astype(str)

    data_copy['day'] = data_copy['day'].astype(str)



    plt.figure(figsize=(30, 10))

    if by == 'year':

        stat = data_copy['year'].value_counts()        

        sns.countplot(by, data=data_copy)

        plt.xlabel(by)

    elif by == 'month':

        data_copy['month'] = data_copy['year'] + '/' + data_copy['month']

        stat = data_copy['month'].value_counts()        

        sns.countplot(by, data=data_copy)

        plt.xlabel(by)

        plt.xticks(rotation=45)

    elif by == 'day':

        data_copy['day'] = data_copy['year'] + '/' + data_copy['month'] + '/' + data_copy['day']

        stat = data_copy['day'].value_counts()        

        sns.countplot(by, data=data_copy)            

        

    plt.ylabel('count')

    plt.title('Count by Year/Month/Day Recent to Old')

    plt.show()

    

    return stat

    
countplot(cleaned, 'year')
countplot(cleaned, 'month')
countplot(cleaned, 'day')
def wordcloud(dataset, min_freq=1):

    bow = list()

    for title in tqdm(dataset['title_cleaned']):

        bow += word_tokenize(title)

    

    word_freq = dict()

    counter = Counter(bow)

    for word, freq in counter.items():

        if freq >= min_freq:

            word_freq[word] = freq

    

#     reddit_mask = np.array(Image.open('/kaggle/working/reddit_icon.png'))    

    

    wc = WordCloud(width=800, height=800, background_color='white') #, mask=reddit_mask)

    wc = wc.generate_from_frequencies(word_freq)

    

    plt.figure(figsize=(12, 12))

    plt.imshow(wc, interpolation="bilinear")

    plt.axis("off")

    plt.show()



    

    return counter, word_freq
counter, word_freq = wordcloud(cleaned)
def wordcloud_by_date(dataset, year=None, month=None, day=None):

    dataset_cp = dataset.copy()

    

    if year:

        dataset_cp = dataset_cp[dataset_cp['year'] == year]

    if month:

        dataset_cp = dataset_cp[dataset_cp['month'] == month]

    if day:

        dataset_cp = dataset_cp[dataset_cp['day'] == day]

    

    return wordcloud(dataset_cp)
counter_2020, word_freq_2020 = wordcloud_by_date(cleaned, year=2020)
counter_20170401, word_freq_20170401 = wordcloud_by_date(cleaned, year=2017, month=4, day=1)
counter_20170402, word_freq_20170402 = wordcloud_by_date(cleaned, year=2017, month=4, day=2)
counter_201704, word_freq_201704 = wordcloud_by_date(cleaned, year=2017, month=4)
counter_04, word_freq_04 = wordcloud_by_date(cleaned, month=4)
def count_vectorize(dataset):

    

    vectorizer = CountVectorizer()

    

    documents = list()

    for title in tqdm(dataset['title_cleaned']):

        documents.append(title)

    document_vector = vectorizer.fit_transform(documents)

    return vectorizer, document_vector
cv, cv_encoded = count_vectorize(cleaned)
cv_encoded.shape
for i, j in zip(cv_encoded.nonzero()[0][:30], cv_encoded.nonzero()[1][:30]):

    print('({:4}, {:8}({:15})) -> {}'.format(i, j, cv.get_feature_names()[j], cv_encoded[i, j]))
def tfidf_vectorize(dataset):

    vectorizer = TfidfVectorizer()

    

    documents = list()

    for title in tqdm(dataset['title_cleaned']):

        documents.append(title)

    document_vector = vectorizer.fit_transform(documents)

    return vectorizer, document_vector
tfidf, tfidf_encoded = tfidf_vectorize(cleaned)
tfidf_encoded.shape
for i, j in zip(tfidf_encoded.nonzero()[0][:30], tfidf_encoded.nonzero()[1][:30]):

    print('({:4}, {:8}({:15})) -> {}'.format(i, j, tfidf.get_feature_names()[j], tfidf_encoded[i, j]))
oc = cleaned[cleaned['original_content'] == 1]

noc = cleaned[cleaned['original_content'] == 0]
def scatter(data, x):

    oc = data[data['original_content'] == 1]

    noc = data[data['original_content'] == 0]



    plt.figure(figsize=(10, 10))

    plt.scatter(x[oc.index, 0], x[oc.index, 1], color='red', label='Original Content')

    plt.scatter(x[noc.index, 0], x[noc.index, 1], color='blue', label='Not Original Content')

    plt.legend()

    plt.title('Sample 2-Dimension Features')

    plt.show()
def svd(encoded, dimension=50):

    svd = TruncatedSVD(n_components=dimension, n_iter=10, random_state=2020)

    reduced = svd.fit_transform(encoded)

    return svd, reduced
svd50, svd50_reduced = svd(tfidf_encoded)
svd50_reduced.shape
scatter(cleaned, svd50_reduced)
def tsne(encoded, dimension=2):

    tsne = TSNE(n_components=dimension, verbose=5, random_state=2020, n_jobs=4)

    reduced = tsne.fit_transform(encoded)

    return tsne, reduced
# tsne2, tsne2_reduced = tsne(svd50_reduced)
# tsne2_reduced.shape
# scatter(cleaned, tsne2_reduced)
X = svd50_reduced

Y = np.array(cleaned['original_content'])
X.shape
Y.shape
x_train, x_test, y_train, y_test = train_test_split(

    X, Y,

    test_size=0.2,

    stratify=Y

)
x_train.shape
x_test.shape
def gradient_boosting_model(x_train, y_train, x_test, y_test):

    model = GradientBoostingClassifier(learning_rate=0.01, n_estimators=100, random_state=2020, verbose=1)

    scores = cross_validate(model, x_train, y_train, scoring='accuracy', cv=2, return_train_score=True, verbose=1)



    model.fit(x_train, y_train)

    acc = accuracy_score(model.predict(x_test), y_test)

    

    return model, scores, acc
gb_model, gb_scores, gb_acc = gradient_boosting_model(x_train, y_train, x_test, y_test)
print('Validation Accuracies: {}'.format(gb_scores['test_score']))

print('Test Accuracy: {}'.format(gb_acc))
def lda(encoded, n_topic=10):

    lda = LatentDirichletAllocation(n_components=n_topic, verbose=1, random_state=2020)

    lda.fit(encoded)

    return lda
lda10 = lda(tfidf_encoded, n_topic=10)
for idx, topic in enumerate(lda10.components_):

    words = [tfidf.get_feature_names()[topic_id] for topic_id in topic.argsort()[::-1][:10]]

    print('Topic {:2d} -> {}'.format(idx, words))
topic_df = cleaned.copy()

length = cleaned['title_cleaned'].shape[0]

for idx, title in tqdm(enumerate(cleaned['title_cleaned'])):

    encoded = tfidf_encoded[idx]

    topics = lda10.transform(encoded)

    topic = topics.argsort()[0][::-1][0]



    topic_df.loc[idx, 'topic'] = topic

    topic_df.loc[idx, 'topic_value'] = topics[0][topic]



    if idx % 30000 == 0 or idx == length - 1:

        print('Topic {:2d}({:.6f}) {:}'.format(topic, topics[0][topic], title))

topic_df
topic_df['topic'] = topic_df['topic'].astype(int)
plt.figure(figsize=(12, 12))

sns.countplot(topic_df['topic'])

plt.xlabel('Topic')

plt.title('Topic Counter')

plt.show()
svd2, svd2_reduced = svd(tfidf_encoded, dimension=2)
svd2_reduced.shape
scatter(cleaned, svd2_reduced)
plt.figure(figsize=(12, 12))

for topic in sorted(topic_df['topic'].unique()):

    index = topic_df[topic_df['topic'] == topic].index

    sns.scatterplot(x=svd2_reduced[index, 0], y=svd2_reduced[index, 1], label=str(topic), s=100)

plt.legend()

plt.title('Topic Distribution in 2-d representation')

plt.show()