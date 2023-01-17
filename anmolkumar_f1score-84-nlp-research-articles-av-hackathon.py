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
import re
import string

# Reset the output dimensions
import matplotlib.pyplot as plt

from sklearn import decomposition
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.metrics import f1_score, accuracy_score, hamming_loss
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer

from skmultilearn.problem_transform import BinaryRelevance, LabelPowerset

from scipy import linalg

from collections import Counter

import pickle

import nltk
nltk.download('wordnet')
from nltk import stem
from nltk.stem import PorterStemmer, WordNetLemmatizer, SnowballStemmer

from gensim import matutils, models
from gensim.models import Word2Vec

import scipy.sparse

from wordcloud import WordCloud


import warnings

plt.rcParams['figure.figsize'] = [24, 12]
plt.style.use('seaborn-darkgrid')
train = pd.read_csv('/kaggle/input/janatahack-independence-day-2020-ml-hackathon/train.csv')
test = pd.read_csv('/kaggle/input/janatahack-independence-day-2020-ml-hackathon/test.csv')
submission = pd.read_csv('/kaggle/input/janatahack-independence-day-2020-ml-hackathon/sample_submission_UVKGLZE.csv')
train.columns = train.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
test.columns = test.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

print('Train Data shape: ', train.shape, 'Test Data shape: ', test.shape)

train.head(10)
def remove_pattern(text, pattern):
    r = re.findall(pattern, text)
    for i in r:
        text = re.sub(i, "", text)
    return text
for column in ['title', 'abstract']:
    #train[column] = train[column].apply(lambda x: x.lower())
    train[column] = np.vectorize(remove_pattern)(train[column], "@[\w]*")
    train[column] = np.vectorize(remove_pattern)(train[column], "#[\w]*")
    train[column] = np.vectorize(remove_pattern)(train[column], '[0-9]')
    train[column] = train[column].str.replace("[^a-zA-Z#]", " ")
    train[column] = train[column].apply(lambda x: ' '.join([i for i in x.split() if len(i) > 3]))

    #test[column] = test[column].apply(lambda x: x.lower())
    test[column] = np.vectorize(remove_pattern)(test[column], "@[\w]*")
    test[column] = np.vectorize(remove_pattern)(test[column], "#[\w]*")
    test[column] = np.vectorize(remove_pattern)(test[column], '[0-9]')
    test[column] = test[column].str.replace("[^a-zA-Z#]", " ")
    test[column] = test[column].apply(lambda x: ' '.join([i for i in x.split() if len(i) > 3]))

train['description'] = train['title'] + " " + train['abstract']
test['description'] = test['title'] + " " + test['abstract']

train.head()
categories = ['computer_science', 'physics', 'mathematics', 'statistics', 'quantitative_biology', 'quantitative_finance']

train_dict = {}

for column in categories:
    a = train.loc[train[column] == 1, 'description'].tolist()
    train_dict[column] = ' '.join(a)
# We can either keep it in dictionary format or put it into a pandas dataframe

data_df = pd.DataFrame(train_dict.items())
data_df.columns = ['index', 'description']
data_df = data_df.set_index('index')
data_df = data_df.sort_index()
data_df.head()
def clean_text(text):
    '''make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
    #text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('[‘’“”…]', '', text)
    text = re.sub('\n', '', text)
    return text
data_df = pd.DataFrame(data_df['description'].apply(lambda x: clean_text(x)))
data_clean = data_df.copy()
data_df.head()
cv = CountVectorizer(stop_words = 'english')
data_cv = cv.fit_transform(data_df['description'])
data_dtm = pd.DataFrame(data_cv.toarray(), columns = cv.get_feature_names())
data_dtm.index = data_df.index
data_dtm = data_dtm.transpose()
data_dtm.head()
# Find the top 30 words on each category

top_dict = {}
for c in data_dtm.columns:
    top = data_dtm[c].sort_values(ascending = False).head(30)
    top_dict[c]= list(zip(top.index, top.values))

top_dict
# Print the top 15 words said by each category

for category, top_words in top_dict.items():
    print(category + ":")
    print(', '.join([word for word, count in top_words[0:14]]))
    print('-----------------------------------------------------------------------------------------------------------------------')
# Let's first pull out the top words for each category

words = []
for category in data_dtm.columns:
    top = [word for (word, count) in top_dict[category]]
    for t in top:
        words.append(t)
        
words

# Let's aggregate this list and identify the most common words along with how many routines they occur in
Counter(words).most_common()
data_dtm
# Find the bottom 200 words on each category

bottom_dict = {}
for c in data_dtm.columns:
    bottom = data_dtm[c].sort_values(ascending = True).head(200)
    bottom_dict[c]= list(zip(bottom.index, bottom.values))

# Let's first pull out the bottom words for each category

bottom_words = []
for category in data_dtm.columns:
    bottom = [word for (word, count) in bottom_dict[category]]
    for b in bottom:
        bottom_words.append(b)
        
bottom_words

# Let's aggregate this list and identify the most common words along with how many routines they occur in
Counter(bottom_words).most_common()
# If less than =2 of the categories have it as a rare word, exclude it from the list

#add_stop_words = [word for word, count in Counter(bottom_words).most_common() if count <= 2]

# Let's update our document-term matrix with the new list of stop words

# Add new stop words

#stop_words = text.ENGLISH_STOP_WORDS.union(add_stop_words)
# If more than 2 of the categories have it as a top word, exclude it from the list

add_stop_words = [word for word, count in Counter(words).most_common() if count > 2]

# Let's update our document-term matrix with the new list of stop words

# Add new stop words

stop_words = text.ENGLISH_STOP_WORDS.union(add_stop_words)

# Recreate document-term matrix

cv = CountVectorizer(stop_words = stop_words)
data_cv = cv.fit_transform(data_clean['description'])
data_stop = pd.DataFrame(data_cv.toarray(), columns = cv.get_feature_names())
data_stop.index = data_clean.index

# Pickle it for later use

pickle.dump(cv, open("cv_stop.pkl", "wb"))
data_stop.to_pickle("dtm_stop.pkl")
data_stop.head()
# Let's make some word clouds!

wc = WordCloud(stopwords = stop_words, background_color = "white", colormap = "Dark2", max_font_size = 150, random_state = 42)
# Create subplots for each category

for index, description in enumerate(data_dtm.columns):
    wc.generate(data_clean.description[description])
    
    plt.subplot(3, 2, index + 1)
    plt.imshow(wc, interpolation = "bilinear")
    plt.axis("off")
    plt.title(categories[index])
    
plt.show()
# Find the number of unique words that each category has

# Identify the non-zero items in the document-term matrix, meaning that the word occurs at least once
unique_list = []
for category in data_dtm.columns:
    uniques = data_dtm[category].to_numpy().nonzero()[0].size
    unique_list.append(uniques)

# Create a new dataframe that contains this unique word count
data_words = pd.DataFrame(list(zip(categories, unique_list)), columns = ['category', 'unique_words'])
data_unique_sort = data_words.sort_values(by = 'unique_words')
data_unique_sort
y_pos = np.arange(len(data_words))

plt.figure(figsize = (8, 8))
plt.barh(y_pos, data_unique_sort.unique_words, align = 'center')
plt.yticks(y_pos, data_unique_sort.category)
plt.title('Number of Unique Words', fontsize = 20)
plt.show()
data = pd.read_pickle('/kaggle/working/dtm_stop.pkl')
data
# One of the required inputs is a term-document matrix
tdm = data.transpose()
tdm.head()
# We're going to put the term-document matrix into a new gensim format, from df --> sparse matrix --> gensim corpus
sparse_counts = scipy.sparse.csr_matrix(tdm)
corpus = matutils.Sparse2Corpus(sparse_counts)
# Gensim also requires dictionary of the all terms and their respective location in the term-document matrix
cv = pickle.load(open("/kaggle/working/cv_stop.pkl", "rb"))
id2word = dict((v, k) for k, v in cv.vocabulary_.items())
# Now that we have the corpus (term-document matrix) and id2word (dictionary of location: term),
# we need to specify two other parameters as well - the number of topics and the number of passes

lda = models.LdaModel(corpus = corpus, id2word = id2word, num_topics = 6, passes = 10)
lda.print_topics()
train_data = train.copy()
train_data = train_data.drop(['title', 'abstract'], axis = 1)
train_data['description'] = train_data['description'].apply(lambda x: clean_text(x))
train['description'] = train['description'].apply(lambda x: clean_text(x))
train_data.head()
train_data, test_data = train_test_split(train_data, random_state = 42, test_size = 0.30, shuffle = True)

trainData = train_data['description'].values.astype('U')
testData = test_data['description'].values.astype('U')

vectorizer = TfidfVectorizer(strip_accents = 'unicode', analyzer = 'word', ngram_range = (1, 3), norm = 'l2', max_features = 10000, use_idf = True, stop_words = stop_words)

#vectorizer = TfidfVectorizer(norm = 'l2', stop_words = stop_words)
vectorizer.fit(trainData)
vectorizer.fit(testData)
X_train = vectorizer.transform(trainData)
y_train = train_data.drop(labels = ['id', 'description'], axis = 1)
X_test = vectorizer.transform(testData)
y_test = test_data.drop(labels = ['id', 'description'], axis = 1)
# Label Powerset

lp_classifier = LabelPowerset(LogisticRegression(max_iter = 250, verbose = 2))
lp_classifier.fit(X_train, y_train)
lp_predictions = lp_classifier.predict(X_test)
print("Accuracy = ", accuracy_score(y_test, lp_predictions))
print("F1 score = ", f1_score(y_test, lp_predictions, average = "micro"))
print("Hamming loss = ", hamming_loss(y_test, lp_predictions))
Test_Data = test['description'].values.astype('U')
Test_Data = vectorizer.transform(Test_Data)
Predictions = lp_classifier.predict(Test_Data)
results = pd.DataFrame.sparse.from_spmatrix(Predictions)
results.columns = tdm.columns.tolist()
results['id'] = test['id'].tolist()
results = results[['id'] + tdm.columns.tolist()]
results.columns = submission.columns.tolist()
results.to_csv('results_v1.csv', index = False)
results.head()
#pipe = Pipeline([('TFidf', TfidfVectorizer(ngram_range = (1, 3), norm = 'l2', stop_words = stop_words, smooth_idf = True)), ("multilabel", OneVsRestClassifier(SVC(kernel = 'poly', random_state = 42)))])

pipe = Pipeline([('TFidf', TfidfVectorizer(ngram_range = (1, 3), norm = 'l2', stop_words = stop_words, smooth_idf = True)), 
                 ("multilabel", MultiOutputClassifier(LinearSVC(penalty = 'l2', random_state = 42, class_weight = 'balanced')))])
y_train = train[[i for i in train.columns if i not in ['id', 'title', 'abstract', 'description']]]
pipe.fit(train['description'], y_train)
predicted1 = pipe.predict(test['description'])
submit = pd.DataFrame({'ID': test['id'].tolist()})
submission = pd.concat([submit, pd.DataFrame(predicted1, columns = tdm.columns.tolist())], axis = 1)
submission.to_csv('submission_F3.csv', index = False)
submission.head()