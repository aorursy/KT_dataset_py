import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import os

from tqdm import tqdm_notebook



from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import BernoulliNB

import eli5

from scipy.sparse import hstack, vstack



from wordcloud import WordCloud, STOPWORDS

from collections import defaultdict

import plotly.graph_objs as go

from plotly import tools

import plotly.offline as py

py.init_notebook_mode(connected=True)

import textblob

import string
print(os.listdir('../input/'))
PATH = '../input/aclimdb/aclImdb/'
train_text = []

test_text = []

train_label = []

test_label = []



for train_test in ['train','test']:

    for neg_pos in ['neg','pos']:

        file_path = PATH + train_test + '/' + neg_pos + '/'

        for file in tqdm_notebook(os.listdir(file_path)):

            with open(file_path + file, 'r') as f:

                if train_test == 'train':

                    train_text.append(f.read())

                    if neg_pos == 'neg':

                        train_label.append(0)

                    else:

                        train_label.append(1)

                else:

                    test_text.append(f.read())

                    if neg_pos == 'neg':

                        test_label.append(0)

                    else:

                        test_label.append(1)
X_train = pd.DataFrame()

X_train['review'] = train_text

X_train['label'] = train_label



X_test = pd.DataFrame()

X_test['review'] = test_text

X_test['label'] = test_label
X_train.head()
# with open(PATH + 'README') as f:

#     readme = f.read()

# readme
sns.countplot(X_train['label']);
X_train[X_train['label']==0]['review'].apply(lambda x: np.log1p(len(x))).hist(alpha=0.6, color='red', label='Negative')

X_train[X_train['label']==1]['review'].apply(lambda x: np.log1p(len(x))).hist(alpha=0.6, color='green', label='Positive')

plt.title('Character count')

plt.legend();
X_train[X_train['label']==0]['review'].apply(lambda x: np.log1p(len(x.split()))).hist(alpha=0.6, color='red', label='Negative')

X_train[X_train['label']==1]['review'].apply(lambda x: np.log1p(len(x.split()))).hist(alpha=0.6, color='green', label='Positive')

plt.title('Word count')

plt.legend();
X_train[X_train['label']==0]['review'].apply(lambda x: np.log1p(len([word for word in x.split() if word.istitle()]))).hist(alpha=0.6, color='red', label='Negative')

X_train[X_train['label']==1]['review'].apply(lambda x: np.log1p(len([word for word in x.split() if word.istitle()]))).hist(alpha=0.6, color='green', label='Positive')

plt.title('Title word count')

plt.legend();
X_train[X_train['label']==0]['review'].apply(lambda x: np.log1p(len([word for word in x.split() if word.isupper()]))).hist(alpha=0.6, color='red', label='Negative')

X_train[X_train['label']==1]['review'].apply(lambda x: np.log1p(len([word for word in x.split() if word.isupper()]))).hist(alpha=0.6, color='green', label='Positive')

plt.title('Upper Case Word count')

plt.legend();
X_train[X_train['label']==0]['review'].apply(lambda x: np.log1p(len(''.join(_ for _ in x if _ in string.punctuation)))).hist(alpha=0.6, color='red', label='Negative')

X_train[X_train['label']==1]['review'].apply(lambda x: np.log1p(len(''.join(_ for _ in x if _ in string.punctuation)))).hist(alpha=0.6, color='green', label='Positive')

plt.title('Punctuation Character count')

plt.legend();
def plot_wordcloud(data, title):

    wordcloud = WordCloud(background_color='black',

                          stopwords=set(STOPWORDS),

                          max_words=200,

                          max_font_size=100,

                          random_state=17,

                          width=800,

                          height=400,

                          mask=None)

    wordcloud.generate(str(data))

    plt.figure(figsize=(15.0,10.0))

    plt.axis('off')

    plt.title(title)

    plt.imshow(wordcloud);
plot_wordcloud(X_train[X_train['label']==0]['review'], 'Negative IMDB reviews')
plot_wordcloud(X_train[X_train['label']==1]['review'], 'Positive IMDB reviews')
X_train_neg = X_train[X_train['label']==0]

X_train_pos = X_train[X_train['label']==1]

additional_stopwords = ['<br', '-', '/><br', '/>the', '/>this']



def generate_ngrams(text, n_gram=1):

    token = [token for token in text.lower().split(' ') if token != '' if token not in STOPWORDS if token not in additional_stopwords]

    ngrams = zip(*[token[i:] for i in range(n_gram)])

    return [' '.join(ngram) for ngram in ngrams]



def horizontal_bar_chart(df, color):

    trace = go.Bar(

        y=df['word'].values[::-1],

        x=df['wordcount'].values[::-1],

        showlegend=False,

        orientation = 'h',

        marker=dict(

            color=color,

        ),

    )

    return trace
freq_dict = defaultdict(int)

for sent in X_train_neg['review']:

    for word in generate_ngrams(sent,1):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ['word', 'wordcount']

trace0 = horizontal_bar_chart(fd_sorted.head(20), 'red')



freq_dict = defaultdict(int)

for sent in X_train_pos['review']:

    for word in generate_ngrams(sent,1):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ['word', 'wordcount']

trace1 = horizontal_bar_chart(fd_sorted.head(20), 'green')



# Creating two subplots

fig = tools.make_subplots(rows=1, cols=2, vertical_spacing=0.04,horizontal_spacing=0.15,

                          subplot_titles=['Frequent unigrams within negative reviews', 

                                          'Frequent unigrams within positive reviews'])

fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 1, 2)

fig['layout'].update(height=600, width=900, paper_bgcolor='rgb(233,233,233)', title='Unigram Count Plots')

py.iplot(fig, filename='word-plots')
freq_dict = defaultdict(int)

for sent in X_train_neg['review']:

    for word in generate_ngrams(sent,2):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ['word', 'wordcount']

trace0 = horizontal_bar_chart(fd_sorted.head(20), 'red')



freq_dict = defaultdict(int)

for sent in X_train_pos['review']:

    for word in generate_ngrams(sent,2):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ['word', 'wordcount']

trace1 = horizontal_bar_chart(fd_sorted.head(20), 'green')



# Creating two subplots

fig = tools.make_subplots(rows=1, cols=2, vertical_spacing=0.04,horizontal_spacing=0.15,

                          subplot_titles=['Frequent bigrams within negative reviews', 

                                          'Frequent bigrams within positive reviews'])

fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 1, 2)

fig['layout'].update(height=600, width=900, paper_bgcolor='rgb(233,233,233)', title='Bigram Count Plots')

py.iplot(fig, filename='word-plots')
X_train_train, X_train_valid, y_train_train, y_train_valid = train_test_split(X_train['review'].values, X_train['label'].values, test_size=0.3, random_state=19)
vect = TfidfVectorizer(ngram_range=(1,1), max_features=50000)

logit = LogisticRegression(C=1, random_state=19)



pipe = Pipeline([

    ('vect', vect),

    ('logit', logit)

])
%%time

pipe.fit(X_train_train, y_train_train)
%%time

preds_valid = pipe.predict(X_train_valid)
print('Accuracy score on the validation dataset: ',accuracy_score(y_train_valid, preds_valid))
_,axes = plt.subplots(figsize=(6,5))

sns.heatmap(confusion_matrix(y_train_valid, preds_valid), annot=True, ax=axes, fmt='g', cmap='Blues', square=True)

axes.set_xlabel('Predicted labels');

axes.set_ylabel('True labels'); 

axes.set_title('Confusion Matrix'); 

axes.xaxis.set_ticklabels(['Negative', 'Positive']);

axes.yaxis.set_ticklabels(['Negative', 'Positive']);
eli5.show_weights(vec=pipe.named_steps['vect'],

                  estimator=pipe.named_steps['logit'],

                  top=10)
vect = TfidfVectorizer(ngram_range=(1,3), max_features=100000)

logit = LogisticRegression(C=1, random_state=19)



pipe = Pipeline([

    ('vect', vect),

    ('logit', logit)

])
%%time

pipe.fit(X_train_train, y_train_train)
%%time

preds_valid = pipe.predict(X_train_valid)
print('Accuracy score on the validation dataset: ',accuracy_score(y_train_valid, preds_valid))
eli5.show_weights(vec=pipe.named_steps['vect'],

                  estimator=pipe.named_steps['logit'],

                  top=10)
vect = CountVectorizer(ngram_range=(1,3), max_features=100000)

logit = LogisticRegression(C=1, random_state=19)



pipe = Pipeline([

    ('vect', vect),

    ('logit', logit)

])
%%time

pipe.fit(X_train_train, y_train_train)
%%time

preds_valid = pipe.predict(X_train_valid)
print('Accuracy score on the validation dataset: ',accuracy_score(y_train_valid, preds_valid))
eli5.show_weights(vec=pipe.named_steps['vect'],

                  estimator=pipe.named_steps['logit'],

                  top=10)
%%time

vect = CountVectorizer(ngram_range=(1,3), max_features=100000)

X_train_train_vect = vect.fit_transform(X_train_train)

X_train_valid_vect = vect.transform(X_train_valid)
X_train_train_vect, X_train_valid_vect
fun1 = lambda x: len(x)

feat1_train = [fun1(review) for review in X_train_train]

feat1_valid = [fun1(review) for review in X_train_valid]



fun2 = lambda x: len(x.split())

feat2_train = [fun2(review) for review in X_train_train]

feat2_valid = [fun2(review) for review in X_train_valid]



fun3 = lambda x: len([word for word in x.split() if word.istitle()])

feat3_train = [fun3(review) for review in X_train_train]

feat3_valid = [fun3(review) for review in X_train_valid]



fun4 = lambda x: len([word for word in x.split() if word.isupper()])

feat4_train = [fun4(review) for review in X_train_train]

feat4_valid = [fun4(review) for review in X_train_valid]



fun5 = lambda x: len(''.join(_ for _ in x if _ in string.punctuation))

feat5_train = [fun5(review) for review in X_train_train]

feat5_valid = [fun5(review) for review in X_train_valid]
X_train_train_custom = hstack([

    X_train_train_vect,

    pd.DataFrame(feat1_train),

    pd.DataFrame(feat2_train),

    pd.DataFrame(feat3_train),

    pd.DataFrame(feat4_train),

    pd.DataFrame(feat5_train)

])

X_train_valid_custom = hstack([

    X_train_valid_vect,

    pd.DataFrame(feat1_valid),

    pd.DataFrame(feat2_valid),

    pd.DataFrame(feat3_valid),

    pd.DataFrame(feat4_valid),

    pd.DataFrame(feat5_valid)

])
X_train_train_custom, X_train_valid_custom
logit = LogisticRegression(C=1, random_state=19)
%%time

logit.fit(X_train_train_custom, y_train_train)
%%time

preds_valid = logit.predict(X_train_valid_custom)
print('Accuracy score on the validation dataset: ',accuracy_score(y_train_valid, preds_valid))
import nltk.stem

from nltk import word_tokenize          

from nltk.stem import WordNetLemmatizer
class LemmaTokenizer(object):

    def __init__(self):

        self.wnl = WordNetLemmatizer()

    def __call__(self, articles):

        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]



lem_vect = CountVectorizer(tokenizer=LemmaTokenizer(),

                           max_features=100000,

                           ngram_range=(1,3),

#                            stop_words='english'

                          )
%%time

X_train_train_vect = lem_vect.fit_transform(X_train_train)

X_train_valid_vect = lem_vect.transform(X_train_valid)
X_train_train_custom = hstack([

    X_train_train_vect,

    pd.DataFrame(feat1_train),

    pd.DataFrame(feat2_train),

    pd.DataFrame(feat3_train),

    pd.DataFrame(feat4_train),

    pd.DataFrame(feat5_train)

])

X_train_valid_custom = hstack([

    X_train_valid_vect,

    pd.DataFrame(feat1_valid),

    pd.DataFrame(feat2_valid),

    pd.DataFrame(feat3_valid),

    pd.DataFrame(feat4_valid),

    pd.DataFrame(feat5_valid)

])
logit = LogisticRegression(C=1, random_state=19)
%%time

logit.fit(X_train_train_custom, y_train_train)
%%time

preds_valid = logit.predict(X_train_valid_custom)
print('Accuracy score on the validation dataset: ',accuracy_score(y_train_valid, preds_valid))
%%time

vect = CountVectorizer(ngram_range=(1,3), max_features=100000)

X_train_train_vect = vect.fit_transform(X_train_train)

X_train_valid_vect = vect.transform(X_train_valid)
X_train_train_custom = hstack([

    X_train_train_vect,

    pd.DataFrame(feat1_train),

    pd.DataFrame(feat2_train),

    pd.DataFrame(feat3_train),

    pd.DataFrame(feat4_train),

    pd.DataFrame(feat5_train)

])

X_train_valid_custom = hstack([

    X_train_valid_vect,

    pd.DataFrame(feat1_valid),

    pd.DataFrame(feat2_valid),

    pd.DataFrame(feat3_valid),

    pd.DataFrame(feat4_valid),

    pd.DataFrame(feat5_valid)

])
# Cs = np.logspace(-3,2,10)

# params = {

#     'C':Cs

# }

# logit = LogisticRegression(random_state=19)

# grid_logit = GridSearchCV(logit, params, cv=5)
# %%time

# grid_logit.fit(X_train_train_custom, y_train_train)
# grid_logit.best_params_
# Cs = np.linspace(0.01,0.16,16)

# params = {

#     'C':Cs

# }

# logit = LogisticRegression(random_state=19)

# grid_logit = GridSearchCV(logit, params, cv=5)
# %%time

# grid_logit.fit(X_train_train_custom, y_train_train)
# grid_logit.best_params_
logit = LogisticRegression(C=0.07, random_state=19)

logit.fit(X_train_train_custom, y_train_train)

preds_valid = logit.predict(X_train_valid_custom)

print('Accuracy score on the validation dataset: ',accuracy_score(y_train_valid, preds_valid))
bern = BernoulliNB()
%%time

bern.fit(X_train_train_custom, y_train_train)
preds_valid = bern.predict(X_train_valid_custom)

print('Accuracy score on the validation dataset: ',accuracy_score(y_train_valid, preds_valid))
alp = np.linspace(0.1,1,10)

params = {

    'alpha':alp

}

bern = BernoulliNB()

grid_bern = GridSearchCV(bern, params, cv=5)

grid_bern.fit(X_train_train_custom, y_train_train)
grid_bern.best_params_
bern = BernoulliNB(alpha=0.1)

bern.fit(X_train_train_custom, y_train_train)

preds_valid = bern.predict(X_train_valid_custom)

print('Accuracy score on the validation dataset: ',accuracy_score(y_train_valid, preds_valid))
%%time

vect = CountVectorizer(ngram_range=(1,3), max_features=100000)

X_train_vect = vect.fit_transform(X_train['review'].values)

X_test_vect = vect.transform(X_test['review'].values)
feat1_train = [fun1(review) for review in X_train['review'].values]

feat1_test = [fun1(review) for review in X_test['review'].values]



feat2_train = [fun2(review) for review in X_train['review'].values]

feat2_test = [fun2(review) for review in X_test['review'].values]



feat3_train = [fun3(review) for review in X_train['review'].values]

feat3_test = [fun3(review) for review in X_test['review'].values]



feat4_train = [fun4(review) for review in X_train['review'].values]

feat4_test = [fun4(review) for review in X_test['review'].values]



feat5_train = [fun5(review) for review in X_train['review'].values]

feat5_test = [fun5(review) for review in X_test['review'].values]
X_train_custom = hstack([

    X_train_vect,

    pd.DataFrame(feat1_train),

    pd.DataFrame(feat2_train),

    pd.DataFrame(feat3_train),

    pd.DataFrame(feat4_train),

    pd.DataFrame(feat5_train)

])

X_test_custom = hstack([

    X_test_vect,

    pd.DataFrame(feat1_test),

    pd.DataFrame(feat2_test),

    pd.DataFrame(feat3_test),

    pd.DataFrame(feat4_test),

    pd.DataFrame(feat5_test)

])
X_train_custom, X_test_custom
logit = LogisticRegression(C=0.07, random_state=19)
%%time

logit.fit(X_train_custom, X_train['label'].values)
preds_test = logit.predict(X_test_custom)

print('Accuracy score on the validation dataset: ',accuracy_score(X_test['label'].values, preds_test))