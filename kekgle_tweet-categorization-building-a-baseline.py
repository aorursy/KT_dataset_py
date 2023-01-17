import numpy  as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.mlab  as mlab



import unicodedata

import spacy

import html

import json

import re



from scipy.stats import norm

from tqdm.autonotebook import tqdm

from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import precision_recall_fscore_support
spacy_nlp = spacy.load('en')
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv', header=0, index_col=0)



pie = {{0: 'Normal Tweets', 1: 'Disaster Tweets'}[k]: v for k, v in dict(train.target.value_counts()).items()}

plt.pie(pie.values(), labels=pie.keys( ), autopct='%1.1f%%', shadow=True, startangle=30, explode=[5e-02, 0.0])

plt.show()
samples = {}



for _id in tqdm(train.index):

    

    tweet = train.loc[train.index == _id].text.values[ 0 ]

    label = train.loc[train.index == _id].target.values[0]

    

    if tweet not in samples:

        samples[tweet] = {

            'appears': {

                0: 0,

                1: 0

            },

            'ids': [ ]

        }

        

    samples[tweet]['appears'][label] += 1

    samples[tweet]['ids'].append(_id)



duplicates = {k: v for k, v in filter(lambda x: x[1]['appears'][0] + x[1]['appears'][1] != 1, samples.items())}



print(json.dumps(duplicates, indent=2))
relabel = {

    881: 0,

    1723: 0,

    1752: 0,

    1760: 0,

    4068: 1,

    4656: 0,

    5662: 1,

    5996: 1,

    6012: 0,

    6087: 0,

    6088: 0,

    6094: 0,

    6113: 0,

    6220: 0,

    6537: 1,

    8018: 0,

    8698: 0,

    9470: 0

}



for k, v in relabel.items():

    train.loc[train.index == k, 'target'] = v



rows_to_drop = {_id for _id, row in train.iterrows() if row['text'] in duplicates and _id not in relabel}

rows_to_keep = {_id for duplicate, info in duplicates.items() for _id in info['ids'][:1]}   # keep first.

rows_to_drop = rows_to_drop.difference(rows_to_keep)



# let's make sure we have not removed one-too-many sample (we only want the duplicates to be gone...)



assert sum([v['appears'][0] + v['appears'][1] for v in duplicates.values()]) - len(duplicates.keys()) == len(rows_to_drop)



for _id in rows_to_drop:

    train.drop(_id, inplace=True)
train.loc[train.target == 1].head(100)
train.loc[train.target == 0].head(100)
def side_by_side_histograms(arr0, arr1, title0, title1, buckets=20):

    """

    Plot two arrays as histograms side by side, with

    normal curve fit on-top.

    """

    mu0, std0 = norm.fit(arr0)

    mu1, std1 = norm.fit(arr1)

    

    # two (2) subplots in one line (1) - histograms will be displayed side-by-side

    

    fig, axs = plt.subplots(1, 2)

    

    # adjusting the spacing between the subplots so they do not end up overlapping

    

    plt.subplots_adjust(wspace=1, left=0.1, right=1.9)

    

    _, bins0, _ = axs[0].hist(arr0, bins=buckets, normed=True)

    _, bins1, _ = axs[1].hist(arr1, bins=buckets, normed=True)

    

    y0 = mlab.normpdf(bins0, mu0, std0)

    axs[0].plot(bins0, y0, 'r--', linewidth=2)



    y1 = mlab.normpdf(bins1, mu1, std1)

    axs[1].plot(bins1, y1, 'r--', linewidth=2)



    

    axs[0].set_title(title0)

    axs[1].set_title(title1)

    

    axs[0].set_ylabel('prob')

    axs[1].set_ylabel('prob')

    

    axs[0].set_xlabel('# chars')

    axs[1].set_xlabel('# token')



    plt.show()
def get_tweets_length(tweets):

    tokens_, lengths = [], []

    for tweet in tqdm(tweets):

        tokns = [tok for tok in spacy_nlp(tweet) if not tok.is_punct]

        lengths.append(len(tweet))

        tokens_.append(len(tokns))

    

    return tokens_, lengths
side_by_side_histograms(

    *get_tweets_length(train.loc[train.target == 1].text.values), 

    'Disaster Tweets: Token Number Distribution',

    'Disaster Tweets: Tweet Length Distribution'

)  
side_by_side_histograms(

    *get_tweets_length(train.loc[train.target == 0].text.values), 

    'Normal Tweets Token Number Distribution',

    'Normal Tweets Tweet Length Distribution'

) 
def extract_hashtags(tweet):

    return ", ".join([  # we return the hashtags in a tweet split by commas

        match.group(0)[1:] for match in re.finditer(r'[#][a-zA-Z]+', tweet)

    ]) or np.nan





train['hashtag'] = train['text'].apply(lambda t: extract_hashtags(t))

train = train[ ['location', 'keyword', 'hashtag', 'text', 'target'] ]

train.head(10)  # et voilà! hashtags are now part of our dataframe:
disaster_hashtags = [ht for ht in train.loc[train.target == 1].hashtag.values]

regulare_hashtags = [ht for ht in train.loc[train.target == 0].hashtag.values]



# let's remember to separate the hashtags by splitting them on commas!



n_disaster_hashtags_by_tweet = [len(ht.split(', ')) if isinstance(ht, str) else 0 for ht in disaster_hashtags]

n_regulare_hashtags_by_tweet = [len(ht.split(', ')) if isinstance(ht, str) else 0 for ht in regulare_hashtags]



side_by_side_histograms(

    n_disaster_hashtags_by_tweet,

    n_regulare_hashtags_by_tweet,

    'Hashtags Per Disaster Tweet',

    'Hashtags Per Regular  Tweet',

    buckets=10

)
train[train.keyword.notna() & train.location.notna()].head(100)
train['keyword'] = train['keyword'].map(lambda s: s.replace('%20', ' ') if isinstance(s, str) else s)



unique_keywords  = {kw for kw in train['keyword' ].values if isinstance(kw, str)}

unique_locations = {lc for lc in train['location'].values if isinstance(lc, str)}



print(f'unique keywords:  {len(unique_keywords)}  (out of {len(train) - len(train[train["keyword" ].isna()])})')

print(f'unique locations: {len(unique_locations)} (out of {len(train) - len(train[train["location"].isna()])})')

print('------------')

print(f'* {len(train[train["keyword" ].isna()])}\t samples have no keywords.')

print(f'* {len(train[train["location"].isna()])}\t samples have no location.')
disaster_keywords = [kw for kw in train.loc[train.target == 1].keyword]

regulare_keywords = [kw for kw in train.loc[train.target == 0].keyword]





def count_values(values):

    return dict(pd.DataFrame(data={'x': values}).x.value_counts())





disaster_keywords_counts = count_values(disaster_keywords)

regulare_keywords_counts = count_values(regulare_keywords)



all_keywords_counts = count_values( train.keyword.values )



# we sort the keywords so the most frequents are on top and we print them with relative

# occurrences in both classes of tweets:



for keyword, _ in sorted(all_keywords_counts.items(), key=lambda x: x[1], reverse=True)[:20]:

    print(f'* {keyword}: ')

    print(f'  - occurrences in disaster tweets: {disaster_keywords_counts.get(keyword, 0)}')

    print(f'  - occurrences in regular  tweets: {regulare_keywords_counts.get(keyword, 0)}')

    print('----')
# since hashtags can be written in many different ways (capitalized, lowercased, camelcased), we lower-

# case them all to group together as many spelling variations as possible in our analysis (again, reme-

# mber to split the hashtags by commas, or we are going to get the list of hashtag per tweet as a csv!)



disaster_hashtags = [ht.lower() for tg in train.loc[train.target == 1].hashtag if isinstance(tg, str) for ht in tg.split(', ')]

regulare_hashtags = [ht.lower() for tg in train.loc[train.target == 0].hashtag if isinstance(tg, str) for ht in tg.split(', ')]



disaster_hashtags_counts = count_values(disaster_hashtags)

regulare_hashtags_counts = count_values(regulare_hashtags)



all_hashtags_counts = {

    k: disaster_hashtags_counts.get(k, 0) + 

       regulare_hashtags_counts.get(k, 0)

    

    for k in set(disaster_hashtags).union(set(regulare_hashtags))

}



for hashtag, _ in sorted(all_hashtags_counts.items(), key=lambda x: x[1], reverse=True)[:20]:

    print(f'* #{hashtag}: ')

    print(f'  - occurrences in disaster tweets: {disaster_hashtags_counts.get(hashtag, 0)}')

    print(f'  - occurrences in regular  tweets: {regulare_hashtags_counts.get(hashtag, 0)}')

    print('----')
# prepare the data:

X = train.text.values

y = train.target.values
def sklearn_stratified_kfold(X, y, vect, model, k=5, seed=100):

    

    kfold = StratifiedKFold(n_splits=k, random_state=seed)

    folds = {

        'pre': [],

        'rec': [],

        'f1s': []

    }

    

    pbar = tqdm(total=k)



    for train_index, test_index in kfold.split(X, y):



        # fit only on train data

        vect.fit(X[train_index])

        # turn tweets to vectors

        X_ = vect.transform( X )



        X_train, y_train = X_[train_index], y[train_index]

        X_test,  y_test  = X_[test_index ], y[test_index ]

        model.fit( X_train, y_train )



        y_hat = model.predict(X_test)



        pre, rec, f1s, _ = precision_recall_fscore_support(y_test, y_hat, average='binary')

        folds['pre'].append(pre)

        folds['rec'].append(rec)

        folds['f1s'].append(f1s)

        

        pbar.update(1)

        

    pbar.close()

    

    results = pd.DataFrame(data=folds)

    print(

        results.head(k), '\n\n',

        results.pre.mean(), 

        results.rec.mean(), 

        results.f1s.mean()

    )
SEED = 1000

vect = TfidfVectorizer( ngram_range=(1, 1) )  # run on words

lreg = LogisticRegression(solver='lbfgs', random_state=SEED)



sklearn_stratified_kfold( X, y, vect, lreg, k=5, seed=SEED )
def replace_links(tweet, **kwargs):

    return re.sub(r'http[s]?://[^\s]+', 'link', tweet)



def remove_hashes(tweet, **kwargs):

    return re.sub(r'[#]([a-zA-Z]+)', r'\1', tweet)



def delete_tagged(tweet, **kwargs):

    return re.sub(r'@\w+', '', tweet)



def normalize_text(text, **kwargs):

    text = unicodedata.normalize('NFD', text)

    return re.sub(r'[^\x00-\x7f]', ' ', text)



def unescape_html(tweet, **kwargs):

    return html.unescape(  tweet  )



def process_tokens(tweet, lower=True, remove_stopwords=False, stem=False, lemmatize=False):

    """

    Process the text using spacy, cleaning contracted form, removing stopwords and

    stemming tokens (if specified). Tokens are lower-cased and punctuation removed.

    """

    tokenized_tweet = [

        token

        for token in spacy_nlp( tweet )  if  (

            token.is_alpha or

            token.tag_ in ['VBZ', 'MD', 'VBP']



        )

    ]



    if remove_stopwords:

        # remove stopwords:

        tokenized_tweet = filter(lambda t: not t.is_stop, tokenized_tweet)



    if lemmatize:

        tokenized_tweet = [token.lemma_ for token in tokenized_tweet]

    else:

        tokenized_tweet = [token.text   for token in tokenized_tweet]



    if stem:

        # stemming tokens:

        tokenized_tweet = [

            stemmer.stem(token)  for 

            token in tokenized_tweet

        ]

    

    return (

        ' '.join(tokenized_tweet).lower() if lower else 

        ' '.join(tokenized_tweet)

    )





class CleaningPipeline(object):

    

    def __init__(self, *funcs):

        self.pipeline = funcs

    

    def run(self, text, **kwargs):

        for f in self.pipeline:

            text = f(text, **kwargs)

        

        return text
my_cleaning_pipeline = CleaningPipeline(

    replace_links, 

    remove_hashes,

    delete_tagged,

    unescape_html,

    normalize_text,

    process_tokens,

)
# the line below will take a while to complete... go get a drink or something :)



train['clean'] = train['text'].apply(lambda text: my_cleaning_pipeline.run(text))

train.head(10)
class GloVe(object):

    

    GLOVE_FILE = '../input/glove840b300dtxt/glove.840B.300d.txt'

    

    def __init__(self, load_on_init=False):

        

        self.embeddings = None

        

        if load_on_init:

            self._load()

        

    def _load(self):

        

        def get_coefs(word, *arr):

            try:

                return word, np.array(arr, dtype='float32')

            except ValueError as e:

                return '',   np.zeros(300, dtype='float32')

        

        self.embeddings = dict(

            get_coefs(*line.split(' ')) for line in open(

                self.GLOVE_FILE, 

                #encoding='latin'

            )

        )

            

    def word_embedding(self, word):

        return self.embeddings.get(word, np.zeros(300, dtype='float32'))

    

    def sent_embedding(self, sent):

        """

        Computes sentence embedding by pooling the sen-

        tence's words' embeddings (computes their mean)

        """

        return np.mean([self.word_embedding(w) for w in sent.split()], axis=0)

    

    def fit(self, X):

        pass

    

    def transform(self, X):

        return np.array([self.sent_embedding(x) for x in ([X] if isinstance(X, str) else X)])
glove = GloVe(load_on_init=True)
X_clean = np.array([text or '<empty>' for text in train.clean.values])
sklearn_stratified_kfold(X_clean, y, glove, LogisticRegression(solver='lbfgs', max_iter=200, random_state=SEED), k=5, seed=SEED)
test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv', header=0, index_col=0)

test['clean'] = test['text'].apply( lambda  text : my_cleaning_pipeline.run(  text  ) )

test.head(10)
X_train = glove.transform(np.array([text or '<empty>' for text in train.clean.values]))

X_test  = glove.transform(np.array([text or '<empty>' for text in test.clean.values ]))
model = LogisticRegression(solver='lbfgs', max_iter=200, random_state=SEED)

model.fit(X_train, y)



y_hat = model.predict(X_test)



y_hat
submission = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')

submission['target'] = y_hat

submission.head(100)
submission.to_csv('submission.csv', index=False)