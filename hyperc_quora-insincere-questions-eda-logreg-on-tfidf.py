import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('display.max_colwidth', -1)
import os
print(os.listdir("../input"))
import re

#Plot
from plotly import tools
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import cufflinks as cf
cf.go_offline()

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.manifold import TSNE

# Keras NLP
from keras.preprocessing import sequence, text

# Keras training
from keras.callbacks import EarlyStopping
from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D

# RNN
from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils

# print
from tqdm import tqdm
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train[train['target']==0].head(5)
train[train['target']==1].head(5)
# remove punc
def remove_punc(question):
    return question.replace("?", "").replace("!", "").replace(",", "").replace(";", "").replace(".", "").replace("\\", "").replace("(", "").replace(")", "")

nb_train = train.shape[0]

print("There are %s questions in the training set." % nb_train)
print("There are %s questions in the testing set.\n" % test.shape[0])

nb_insincere = train[train['target']==1].shape[0]
print("There are {} insincere questions in the training set. ({}% total)".format(nb_insincere, round(100*nb_insincere/nb_train,2)))
def question_size(question):
    return len(question.split(" "))

train['question_size'] = train["question_text"].apply(question_size)
test['question_size'] = test["question_text"].apply(question_size)
# Sampled histograms
to_plot = pd.DataFrame({'train': train['question_size'].sample(frac=0.01),
                   'test': test['question_size'].sample(frac=0.01)})

to_plot.iplot(kind='histogram',
              histnorm='probability',
              title='Train/test questions size distribution (from samples)',
              filename='cufflinks/basic-histogram')
# Sampled histograms
to_plot = pd.DataFrame({'train_insincere': train[train['target']==1]['question_size'].sample(frac=0.05),
                   'train_sincere': train[train['target']==0]['question_size'].sample(frac=0.01)})

to_plot.iplot(kind='histogram',
              histnorm='probability',
              title='Insincere/sincere questions size distribution (from samples)',
              filename='cufflinks/basic-histogram')
#do better here
train['number_questions'] = train["question_text"].apply(lambda x:x.count('? '))
train['number_statements'] = train["question_text"].apply(lambda x:x.count('. '))
#train['number_statements'] = train["question_text"].apply(lambda x:x.count("[^.]{15,}."))
# Sampled histograms
to_plot = pd.DataFrame({'train_insincere': train[train['target']==1]['number_statements'].sample(frac=0.05),
                   'train_sincere': train[train['target']==0]['number_statements'].sample(frac=0.01)})

to_plot.iplot(kind='histogram',
              histnorm='probability',
              title='Train/test, number of statements in questions distribution (from samples)',
              filename='cufflinks/basic-histogram')
punc_list = ['\\', '?', '.', ';', ',', '-']

def average_word_length(question):
    words = re.sub("|".join(punc_list), "", question).split(" ")
    return np.mean([len(w) for w in words])

train['average_word_length'] = train["question_text"].apply(average_word_length)
# Sampled histograms
to_plot = pd.DataFrame({'train_insincere': train[train['target']==1]['average_word_length'].sample(frac=0.01),
                   'train_sincere': train[train['target']==0]['average_word_length'].sample(frac=0.01)})

to_plot.iplot(kind='histogram', histnorm='probability',
              title='Train/test, average word length in questions distribution (from samples)', filename='cufflinks/basic-histogram')
import nltk
from nltk.corpus import stopwords
eng_stopwords = set(stopwords.words("english"))

def nb_stop_words(question):
    words = re.sub("|".join(punc_list), "", question).split(" ")
    return len([w for w in words if w in eng_stopwords])

train['nb_stop_words'] = train["question_text"].apply(nb_stop_words)
# Sampled histograms
to_plot = pd.DataFrame({'train_insincere': train[train['target']==1]['nb_stop_words'].sample(frac=0.01),
                   'train_sincere': train[train['target']==0]['nb_stop_words'].sample(frac=0.01)})

to_plot.iplot(kind='histogram', histnorm='probability',
              title='Train/test, number of stopwords in questions distribution (from samples)', filename='cufflinks/basic-histogram')
def nb_up_case_word(question):
    return question.count()

train['nb_up_case_word'] = train["question_text"].apply(lambda x:len(re.findall('[A-Z][a-z]+ ',x)))
# Sampled histograms
to_plot = pd.DataFrame({'train_positive': train[train['target']==1]['nb_up_case_word'].sample(frac=0.01),
                   'train_negative': train[train['target']==0]['nb_up_case_word'].sample(frac=0.01)})

to_plot.iplot(kind='histogram', histnorm='probability',
              title='Train/test, number of First names in questions distribution (from samples)', filename='cufflinks/basic-histogram')
def get_question_type(question):
    question = question.lower()
    question_types = ["why", "who", "what", "when", "how", "can", "do"]
    
    for qt in question_types:
        if question.startswith(qt):
            return qt
    
    return "other"

train["question_type"] = train["question_text"].apply(get_question_type)
train["sum"] = train.groupby(["target"]).qid.count()
train_grouped = train.groupby(["question_type", "target"],as_index=False).qid.count()

train_grouped["sum"] = train_grouped.groupby("target").qid.transform(np.sum)
train_grouped["qid"] = train_grouped["qid"] / train_grouped["sum"]

train_pivoted = train_grouped.pivot(index="question_type", columns="target", values="qid")
train_pivoted.iplot(kind='bar',
              title='Train/test, type of questions distribution (from samples)', filename='cufflinks/bar-chart-row')
#Build vocabulay
ctv = CountVectorizer(analyzer='word',token_pattern=r"\w{1,}'?[t]?", ngram_range=(1, 1), stop_words = 'english')

# Get vocabulary
train['question_text_no_punc'] = train['question_text'].apply(remove_punc)
ctv.fit(train['question_text_no_punc'])

train_pos = ctv.transform(train[train['target']==1]['question_text_no_punc'])
train_neg = ctv.transform(train[train['target']==0]['question_text_no_punc'])
train_pos = train_pos.sum(axis=0) / train_pos.sum()
train_neg = train_neg.sum(axis=0) / train_neg.sum()

train_diff = train_pos - train_neg

train_diff = train_diff.tolist()[0]
inv_voc = {v:k for k,v in ctv.vocabulary_.items()}
tup_list = [(value, inv_voc[ind]) for (ind, value) in enumerate(train_diff)]
most_sincere_words = sorted(tup_list)
most_insincere_words = sorted(tup_list, reverse=True)
[pos_scores, pos_words] = zip(*most_insincere_words)
[neg_scores, neg_words] = zip(*most_sincere_words)

trace1 = go.Bar(
    y=list(reversed(pos_words[:300])),
    x=list(reversed(pos_scores[:300])),
    orientation = 'h',
    marker=dict(
        color='rgba(244, 80, 65, 0.6)'
    ),
)
trace2 = go.Bar(
    y=list(reversed(neg_words[:300])),
    x=list(reversed(neg_scores[:300])),
    orientation = 'h',
    marker=dict(
        color='rgba(134, 244, 66, 0.6)'
    ),
)

fig = tools.make_subplots(rows=1, cols=2)

fig.append_trace(trace2, 1, 1)
fig.append_trace(trace1, 1, 2)

fig['layout'].update(height=8000, title='Words that are more used in sincere(left)/insincere(right) questions')
py.iplot(fig, filename='simple-subplot-with-annotations')
# user embeddings to find words that don't exist
EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt' # https://www.kaggle.com/sudalairajkumar/a-look-at-different-embeddings
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))
def nb_typos(question):
    real_words = embeddings_index.keys()
    typos=[w for w in remove_punc(question.lower()).split(" ") if w not in real_words]
    
    return len(typos)
    
train["nb_typos"] = train["question_text"].apply(nb_typos)
# Sampled histograms
to_plot = pd.DataFrame({'train_positive': train[train['target']==1]['nb_typos'].sample(frac=0.01),
                   'train_negative': train[train['target']==0]['nb_typos'].sample(frac=0.01)})

to_plot.iplot(kind='histogram', histnorm='probability',
              title='Train/test, number of typos in questions distribution (from samples)', filename='cufflinks/basic-histogram')
# match and highlight in 2D
EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

top_sincere_words = most_sincere_words[:1000]
top_insincere_words = most_insincere_words[:1000]
words_to_keep = top_sincere_words + top_insincere_words
emb = np.array([list(embeddings_index[word[1]]) + [abs(word[0])/word[0], word[1]] for word in words_to_keep if word[1] in embeddings_index ])
emb.shape
X = emb
X_embedded = TSNE(n_components=2).fit_transform(X[:,:300])
X_embedded.shape
# Create a trace
trace = go.Scatter(
    x = X_embedded[:,0],
    y = X_embedded[:,1],
    mode = 'markers',
    marker=dict(
        color= ['rgb(51, 206, 111)' if val=='-1.0' else 'rgb(244, 86, 66)'  for val in list(X[:,300])]
    ),
    text = X[:,301]
)

data = [trace]
layout = go.Layout(title='top sincere/insincere words according to their embeddings (projected on 2D by T-SNE)')
fig = go.Figure(data=data, layout=layout)
# Plot and embed in ipython notebook!
py.iplot(fig, filename='basic-scatter')

# or plot with: plot_url = py.plot(data, filename='basic-line')
train_count = train.groupby('target').count()
train_count.qid[0] / (train_count.qid[0] + train_count.qid[1])
xtrain, xvalid, ytrain, yvalid = train_test_split(train.question_text, train.target, 
                                                  stratify=train.target, 
                                                  random_state=42, 
                                                  test_size=0.1, shuffle=True)


tfv = TfidfVectorizer(min_df=3,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')

# Fitting TF-IDF to both training and test sets (semi-supervised learning)
tfv.fit(list(xvalid) + list(xtrain))

train_tfv =  tfv.transform(xtrain)
valid_tfv =  tfv.transform(xvalid)
train_tfv
valid_tfv
# Fitting a simple Logistic Regression on TFIDF
clf = LogisticRegression(C=1.0)
clf.fit(train_tfv, ytrain)
predictions = clf.predict_proba(valid_tfv)
# Plot different F1 scores according to threshold
x = np.linspace(0,1, num=25)
def pred(threshold):
    return [0 if (y<threshold) else 1 for y in predictions[:,1]]

scores = [f1_score(yvalid, pred(xx)) for xx in x]

# Create a trace
trace = go.Scatter(
    x = x,
    y = scores
)

data = [trace]

py.iplot(data, filename='basic-line')
ctv = CountVectorizer(analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3), stop_words = 'english')

# Fitting Count Vectorizer to both training and test sets (semi-supervised learning)
ctv.fit(list(xtrain) + list(xvalid))
xtrain_ctv =  ctv.transform(xtrain) 
xvalid_ctv = ctv.transform(xvalid)
# Fitting a simple Logistic Regression on TFIDF
clf = LogisticRegression(C=1.0)
clf.fit(xtrain_ctv, ytrain)
predictions = clf.predict_proba(xvalid_ctv)
# Plot different F1 scores according to threshold
x = np.linspace(0,1, num=35)
def pred(threshold):
    return [0 if (y<threshold) else 1 for y in predictions[:,1]]

scores = [f1_score(yvalid, pred(xx)) for xx in x]

# Create a trace
trace = go.Scatter(
    x = x,
    y = scores
)

data = [trace]

py.iplot(data, filename='basic-line')
# Reload embeddings even if already loaded (for each section to be independant)
EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

xtrain, xvalid, ytrain, yvalid = train_test_split(train.question_text, train.target, 
                                                  stratify=train.target, 
                                                  random_state=42, 
                                                  test_size=0.1, shuffle=True)

# using keras tokenizer here
token = text.Tokenizer(num_words=None)
max_len = 70

token.fit_on_texts(list(xtrain) + list(xvalid))
xtrain_seq = token.texts_to_sequences(xtrain)
xvalid_seq = token.texts_to_sequences(xvalid)

# zero pad the sequences
xtrain_pad = sequence.pad_sequences(xtrain_seq, maxlen=max_len)
xvalid_pad = sequence.pad_sequences(xvalid_seq, maxlen=max_len)

word_index = token.word_index
# we need to binarize the labels for the neural net
ytrain_enc = np_utils.to_categorical(ytrain)
yvalid_enc = np_utils.to_categorical(yvalid)
# create an embedding matrix for the words we have in the dataset
embedding_matrix = np.zeros((len(word_index) + 1, 300))
for word, i in tqdm(word_index.items()):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
# A simple LSTM with glove embeddings and two dense layers
model = Sequential()
model.add(Embedding(len(word_index) + 1,
                     300,
                     weights=[embedding_matrix],
                     input_length=max_len,
                     trainable=False))
model.add(SpatialDropout1D(0.3))
model.add(LSTM(100, dropout=0.3, recurrent_dropout=0.3))

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.8))

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.8))

model.add(Dense(2))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(xtrain_pad, y=ytrain_enc, batch_size=512, epochs=10, verbose=1, validation_data=(xvalid_pad, yvalid_enc))
predictions = model.predict_proba(xvalid_pad)
# Plot different F1 scores according to threshold
x = np.linspace(0,1, num=100)
def pred(threshold):
    return [0 if (y<threshold) else 1 for y in predictions[:,1]]

scores = [f1_score(yvalid, pred(xx)) for xx in x]

# Create a trace
trace = go.Scatter(
    x = x,
    y = scores
)

data = [trace]

py.iplot(data, filename='basic-line')
