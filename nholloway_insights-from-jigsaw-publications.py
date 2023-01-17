import gc

import numpy as np

import pandas as pd

import markovify as mk

import plotly



import plotly.offline as py

import plotly.graph_objs as go

from matplotlib import pyplot as plt

from gensim.models import KeyedVectors

from wordcloud import WordCloud, STOPWORDS

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score, train_test_split

from scipy.sparse import hstack

# from keras.preprocessing import text, sequence



plt.style.use('fivethirtyeight') 

plotly.tools.set_credentials_file(username='nholloway', api_key='Ef8vuHMUdvaIpvtC2lux')

py.init_notebook_mode(connected=True)

%matplotlib inline 
train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')

description = pd.DataFrame(index=['observations(rows)', 'percent missing', 'dtype', 'range'])

numerical = []

categorical = []

for col in train.columns:

    obs = train[col].size

    p_nan = round(train[col].isna().sum()/obs, 2)

    num_nan = f'{p_nan}% ({train[col].isna().sum()}/{obs})'

    dtype = 'categorical' if train[col].dtype == object else 'numerical'

    numerical.append(col) if dtype == 'numerical' else categorical.append(col)

    rng = f'{len(train[col].unique())} labels' if dtype == 'categorical' else f'{train[col].min()}-{train[col].max()}'

    description[col] = [obs, num_nan, dtype, rng]



pd.set_option('display.max_columns', 150)

display(description)

display(train.head())
def preprocess(text):

    s_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }

    specials = ["’", "‘", "´", "`"]

    p_mapping = {"_":" ", "`":" "}  

    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

    

    for s in specials:

        text = text.replace(s, "'")

    text = ' '.join([s_mapping[t] if t in s_mapping else t for t in text.split(" ")])

    for p in p_mapping:

        text = text.replace(p, p_mapping[p])    

    for p in punct:

        text = text.replace(p, f' {p} ')     

    return text
IDENTITY_COLUMNS = ['black', 'white', 'male', 'female', 'homosexual_gay_or_lesbian',

                   'christian', 'jewish', 'muslim', 'psychiatric_or_mental_illness'] 

train.fillna(0, inplace=True)

train['target'] = np.where(train['target'] >= 0.5, 1, 0)

train['comment_text'] = train['comment_text'].apply(lambda x: preprocess(x.lower()))

subgroup_df = pd.DataFrame(columns = ['identity', 'toxic', 'non_toxic', 'toxic_ratio', 'percent_of_total'])               

t_counts = train['target'].value_counts()  

num_comments = train['comment_text'].size



for identity in IDENTITY_COLUMNS:

    subgroup = train.loc[(train[identity] != 0.0)]

    subgroup_counts = subgroup['target'].value_counts()

    

    toxic_ratio = round(subgroup_counts[1]/subgroup_counts[0], 2)

    pct_of_total = round(subgroup['comment_text'].size/num_comments, 2)

    subgroup_dict = {'identity': identity, 'toxic': subgroup_counts.iloc[1], 'non_toxic': subgroup_counts.iloc[0], 'toxic_ratio': toxic_ratio,'percent_of_total': pct_of_total}

    subgroup_df = subgroup_df.append(subgroup_dict, ignore_index=True)
trace1 = go.Bar(

    x = subgroup_df['identity'].values.tolist(),

    y = subgroup_df['non_toxic'].values.tolist(),

    name='Non-Toxic Comments'

)



trace2 = go.Bar(

    x = subgroup_df['identity'].values.tolist(),

    y = subgroup_df['toxic'].values.tolist(),

    name='Toxic Comments'

)



data = [trace1, trace2]

layout = go.Layout(

    title='Toxic v. Non-Toxic Comments by Subgroup',

    barmode='stack'

)



fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='stacked')
t_dict = {'identity':'total_comments', 'toxic': t_counts.iloc[1], 'non_toxic': t_counts.iloc[0], 'toxic_ratio': round(t_counts.iloc[1]/t_counts.iloc[0], 2),'percent_of_total': 1.00}

subgroup_df = subgroup_df.append(t_dict, ignore_index=True)



trace1 = go.Bar(

    x = subgroup_df['identity'].values.tolist(),

    y = subgroup_df['non_toxic'].values.tolist(),

    name='Non-Toxic Comments'

)



trace2 = go.Bar(

    x = subgroup_df['identity'].values.tolist(),

    y = subgroup_df['toxic'].values.tolist(),

    name='Toxic Comments'

)



data = [trace1, trace2]

layout = go.Layout(

    title='Toxic v. Non-Toxic Comments by Subgroup',

    barmode='stack'

)



fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='stacked')
display(subgroup_df.T)
comment_len = [len(text) for text in train['comment_text'].values.tolist()]



plt.hist(comment_len, bins=1000)

plt.title('Distribution of Comment Length')

plt.xlabel('Number of Words in Comment')

plt.ylabel('Amount of Comments')

plt.xlim(0, 1100)

plt.show()
GLOVE_PATH = '../input/glove840b300dtxt/glove.840B.300d.txt'



def build_vocab(text):

    sentences = text.apply(lambda x: x.split()).values

    vocab = {}

    for sentence in sentences:

        for word in sentence:

            try:

                vocab[word] += 1

            except KeyError:

                vocab[word] = 1

    return vocab



def get_coefs(word, *arr):

    return word, np.asarray(arr, dtype='float32')



def load_embeddings(path):

    with open(path) as f:

        return dict(get_coefs(*line.strip().split(' ')) for line in f)



def check_coverage(word_index, path):

    known_words = {}

    unknown_words = {}

    embedding_known = 0

    embedding_unknown = 0

    embedding_index = load_embeddings(path)



    for word in word_index.keys():

        try:

            known_words[word] = embedding_index[word]

            embedding_known += word_index[word]

        except:

            unknown_words[word] = word_index[word]

            embedding_unknown += word_index[word]

            pass

        

    percent_vocab = round(len(known_words) / len(word_index), 3)

    percent_text = round(embedding_known / (embedding_known + embedding_unknown), 3)

    del embedding_index, known_words, unknown_words

    gc.collect()

    return percent_vocab, percent_text

IDENTITY_COLUMNS = ['black', 'white', 'male', 'female', 'homosexual_gay_or_lesbian',

                   'christian', 'jewish', 'muslim', 'psychiatric_or_mental_illness'] 

vocab_df = pd.DataFrame(columns = ['identity', 'vocabulary_size', 'unique_vocabulary', 'text_coverage', 'vocab_coverage'])                     



for identity in IDENTITY_COLUMNS:

    subgroup = train.loc[(train[identity] != 0.0)]

    not_subgroup = train.loc[~(train[identity] != 0.0)]

    vocab = build_vocab(subgroup['comment_text'])

    not_vocab = build_vocab(not_subgroup['comment_text'])

    uniq_vocab = set(vocab).difference(set(not_vocab))

    vocab_coverage, text_coverage = check_coverage(vocab, GLOVE_PATH)

    voc_dict = {'identity': identity, 'vocabulary_size': len(vocab), 'unique_vocabulary': len(uniq_vocab),'text_coverage': text_coverage, 'vocab_coverage': vocab_coverage}

    vocab_df = vocab_df.append(voc_dict, ignore_index=True)

display(vocab_df.T)
def get_tfidf_keywords(documents, document, topn=10, ngram=(1, 2)):

    cvec = CountVectorizer(ngram_range=ngram, max_df=0.75, min_df=1, stop_words='english') 

    cvec_term_doc = cvec.fit_transform(documents)

    tvec = TfidfTransformer(smooth_idf=True, norm = None, use_idf=True)

    feature_names = cvec.get_feature_names()

    tvec.fit(cvec_term_doc)

    tfidf = tvec.transform(cvec.transform(document), copy=False)

    sorted_items = sort_coo(tfidf.tocoo())

    keywords = extract_topn_from_vector(feature_names, sorted_items, topn)

    return keywords



# Sort sparse matrix 

def sort_coo(coo_matrix):

    tuples = zip(coo_matrix.col, coo_matrix.data)

    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)



def extract_topn_from_vector(feature_names, sorted_items, topn):

    sorted_items = sorted_items[:topn]

    score_vals = []

    feature_vals = []

    for idx, score in sorted_items:

        score_vals.append(round(score, 3))

        feature_vals.append(feature_names[idx])

    results= {}

    for idx in range(len(feature_vals)):

        results[feature_vals[idx]]=score_vals[idx]

    return results    
sample_text = train['comment_text'].sample(10000).values.tolist()

sample_subgroup = train.loc[(train['female'] != 0.0)]['comment_text'].sample(1000).values.tolist()

tfidf = get_tfidf_keywords(sample_text, sample_subgroup, 20)

wc = WordCloud(background_color='white')

wc.generate_from_frequencies(tfidf)

plt.imshow(wc)

plt.axis('off')

plt.show()
sample_text = train['comment_text'].sample(10000).values.tolist()

sample_subgroup = train.loc[(train['homosexual_gay_or_lesbian'] != 0.0)]['comment_text'].sample(1000).values.tolist()

tfidf = get_tfidf_keywords(sample_text, sample_subgroup, 30)

wc = WordCloud(background_color='white')

wc.generate_from_frequencies(tfidf)

plt.imshow(wc)

plt.axis('off')

plt.show()
sample_text = train['comment_text'].sample(10000).values.tolist()

sample_subgroup = train.loc[(train['black'] != 0.0)]['comment_text'].sample(100).values.tolist()

tfidf = get_tfidf_keywords(sample_text, sample_subgroup, 30)

wc = WordCloud(background_color='white')

wc.generate_from_frequencies(tfidf)

plt.imshow(wc)

plt.axis('off')

plt.show()
train_orig = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv')

description = pd.DataFrame(index=['observations(rows)', 'percent missing', 'dtype', 'range'])

numerical = []

categorical = []

for col in train_orig.columns:

    obs = train_orig[col].size

    p_nan = round(train_orig[col].isna().sum()/obs, 2)

    num_nan = f'{p_nan}% ({train_orig[col].isna().sum()}/{obs})'

    dtype = 'categorical' if train_orig[col].dtype == object else 'numerical'

    numerical.append(col) if dtype == 'numerical' else categorical.append(col)

    rng = f'{len(train_orig[col].unique())} labels' if dtype == 'categorical' else f'{train_orig[col].min()}-{train_orig[col].max()}'

    description[col] = [obs, num_nan, dtype, rng]



final_results = pd.DataFrame(columns = ['parameters', 'training auc score',

                                       'precision', 'training time', 'parameter tuning time'])



pd.set_option('display.max_columns', 150)

display(description)

display(train_orig.head())
train_orig['comment_text'] = train_orig['comment_text'].apply(lambda x: preprocess(x.lower()))

new_comments = set(train['comment_text'])

orig_comments = set(train_orig['comment_text'])

intersection = new_comments.intersection(orig_comments)

intersection
comment_len = [len(text) for text in train_orig['comment_text'].values.tolist()]



plt.hist(comment_len, bins=1000)

plt.title('Distribution of Comment Length')

plt.xlabel('Number of Words in Comment')

plt.ylabel('Amount of Comments')

plt.xlim(0, 1250)

plt.show()

target_count = train_orig['toxic'].value_counts()

plt.bar([0, .5], target_count, width=.3, tick_label=['non-toxic', 'toxic'])

plt.show()
id_text = train.loc[(train['black'] != 0.0)]

id_len = len(id_text)

control_text = train.loc[~(train['black'] != 0.0)].sample(id_len)

# Use .sample to shuffle the rows

train_text = pd.concat([control_text, id_text]).sample(frac=1)

train_text['black'] = np.where(train_text['black'] >= 0.1, 1, 0)

test_x = train_orig['comment_text']

test_idx = train_orig[['id', 'comment_text']]



train_target = train_text['black']

train_x = train_text['comment_text']

all_text = pd.concat([train_x, test_x])
word_vectorizer = TfidfVectorizer(

    sublinear_tf=True,

    strip_accents='unicode',

    analyzer='word',

    token_pattern=r'\w{1,}',

    stop_words='english',

    ngram_range=(1, 1),

    max_features=10000)

# word_vectorizer.fit(all_text)

# train_word_features = word_vectorizer.transform(train_x)

# val_word_features = word_vectorizer.transform(test_x)



char_vectorizer = TfidfVectorizer(

    sublinear_tf=True,

    strip_accents='unicode',

    analyzer='char',

    stop_words='english',

    ngram_range=(2, 6),

    max_features=50000)

# char_vectorizer.fit(all_text)

# train_char_features = char_vectorizer.transform(train_x)

# val_char_features = char_vectorizer.transform(test_x)
# train_features = hstack([train_char_features, train_word_features])

# test_features = hstack([val_char_features, val_word_features])



# scores = []



# classifier = LogisticRegression(C=0.1, solver='sag')



# cv_score = np.mean(cross_val_score(classifier, train_features, train_target, cv=3, scoring='roc_auc'))

# scores.append(cv_score)

# print(f'CV score: {cv_score}')



# classifier.fit(train_features, train_target)

# preds = classifier.predict_proba(test_features)[:, 1]

# predicted = pd.concat([pd.DataFrame(preds), test_idx], axis=1)

# print(predicted.iloc[:20])

subgroup_comments = train.loc[(train['black'] != 0.0) & (train['target'] == 0)]['comment_text'].tolist()

mk_model = mk.Text(subgroup_comments, state_size=3)

for i in range(5):

    text = mk_model.make_short_sentence(100)

    print(text)