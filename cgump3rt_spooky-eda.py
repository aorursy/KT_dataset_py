import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

import sklearn as sk

import nltk

from textblob import TextBlob as TB, Word
df = pd.read_csv("../input/train.csv")

print('loaded %d samples with %d features' %(df.shape))
print(df.head())

print(df.tail())
df['label'] = sk.preprocessing.LabelEncoder().fit_transform(df.author)

df.head()
df.groupby('author').label.value_counts()
df.groupby('author').text.count().plot.bar(title = 'training set balance');
# calculate different measures of the quantity of information

df['n_sentences'] = df.text.transform(lambda x: len(nltk.sent_tokenize(x)))

df['n_words'] = df.text.transform(lambda x: len(nltk.word_tokenize(x)))

df['text_len'] = df.text.transform(lambda x: len(x))
# first group by author

(df.groupby('author')

 # select the columns we are interested in (note the list inside the []-operator)

 [['n_sentences','n_words','text_len']]

 # we calculate the sum for each column within each author group

 .sum()

 # finally, plot as bar chart in different figures

 .plot.bar(subplots = True, layout = (1,3), figsize = (18,6)));
print(df.n_sentences.value_counts())

df.n_sentences.plot.hist(log = True)

print('out of %d text snippets %d contain more than one sentence' %

      (len(df), (df.n_sentences > 1).sum()))
print(df[df.n_sentences == 9].text.iloc[0])
# initialize count vectorizer

cv = sk.feature_extraction.text.CountVectorizer()

# learn vocabulary and calculate term-document frequncies in one go

X = cv.fit_transform(df.text)

print('learned a vocabulary of size %d' % len(cv.vocabulary_))

print('first 5 terms in vocabulary (ordered alphabetically):')

print(cv.get_feature_names()[:5])

print('shape of term-document matrix is [n_samples, vocabulary_size]: ', X.shape)
Y = sk.preprocessing.label_binarize(df.author, df.author.unique())

print('shape of Y: ',Y.shape)

print('The first 5 rows of Y:')

print(Y[:5,:])

print('class labels: ', df.author.unique())
counts = Y.T * X

print('shape of counts is [n_classes, vocabulary_size]: ', counts.shape)
count_df = pd.DataFrame(data = counts.T,

                        columns = df.author.unique(),

                        index = cv.get_feature_names())

count_df.head(10)
_, (ax1, ax2, ax3) = plt.subplots(ncols = 3, figsize = (16,6))

topn = 10

count_df.EAP.sort_values(ascending = False).iloc[topn::-1].plot.barh(title = 'EAP', ax = ax1)

count_df.HPL.sort_values(ascending = False).iloc[topn::-1].plot.barh(title = 'HPL', ax = ax2)

count_df.MWS.sort_values(ascending = False).iloc[topn::-1].plot.barh(title = 'MWS', ax = ax3);
english_sw = nltk.corpus.stopwords.words('english')

print('loaded %d stopwords for english' % len(english_sw))
count_df_no_sw = count_df[~count_df.index.isin(english_sw)]

_, (ax1, ax2, ax3) = plt.subplots(ncols = 3, figsize = (16,6))

topn = 10

count_df_no_sw.EAP.sort_values(ascending = False).iloc[topn::-1].plot.barh(title = 'EAP', ax = ax1)

count_df_no_sw.HPL.sort_values(ascending = False).iloc[topn::-1].plot.barh(title = 'HPL', ax = ax2)

count_df_no_sw.MWS.sort_values(ascending = False).iloc[topn::-1].plot.barh(title = 'MWS', ax = ax3);
cv2 = sk.feature_extraction.text.CountVectorizer(analyzer = 'char')

X2 = cv2.fit_transform(df.text)

char_counts = pd.DataFrame(data = (X2.T * Y),

                           columns = df.author.unique(),

                           index = cv2.get_feature_names())
char_counts
char_counts /= char_counts.sum()

char_counts.head(10)
char_counts = (char_counts.stack()

               .reset_index()

               .rename(columns = {'level_0': 'char',

                                  'level_1': 'author',

                                  0: 'probability'

                                 }))
char_counts.head(10)
sns.factorplot(col = 'char', x = 'author', y = 'probability',

               data = char_counts[char_counts.probability > 1e-5], kind = 'bar', col_wrap = 4, size = 3, log = True);
def text2POS(text):

    tags = TB(text).tags

    return ' '.join([t for _,t in tags])
df['tags'] = df.text.transform(text2POS)
c = sk.feature_extraction.text.CountVectorizer(tokenizer = lambda x: x.split())

X3 = c.fit_transform(df.tags)

a = pd.DataFrame(data = X3.T * Y, columns = df.author.unique(), index = c.get_feature_names())
X3.todense()[0,:]
df.tags.iloc[0]
b = a / a.sum()
b = b.stack().reset_index().rename(columns = {'level_0': 'POS', 'level_1': 'author', 0: 'probability'})
a.div(a.max(axis = 1), axis = 'index').min(axis = 1)
stop_tags = a.index[a.div(a.max(axis = 1), axis = 'index').min(axis = 1) > 0.9]

stop_tags
sns.factorplot(col = 'POS', x = 'author', y = 'probability',

               data = b, kind = 'bar', col_wrap = 4, size = 3, log = True);
from sklearn.pipeline import make_pipeline, make_union

from sklearn.naive_bayes import MultinomialNB



reference = make_pipeline(

    make_union(

        make_pipeline(

            sk.preprocessing.FunctionTransformer(func = lambda df: df.text, validate = False),

            make_union(

                sk.feature_extraction.text.CountVectorizer(min_df = 5),

                sk.feature_extraction.text.CountVectorizer(analyzer = 'char', min_df = 10),

            )

        ),

        make_pipeline(

             sk.preprocessing.FunctionTransformer(func = lambda df: df.lemma, validate = False),

             sk.feature_extraction.text.CountVectorizer(tokenizer = lambda x: x.split(),

                                                        #stop_words = ['cc','nns'],

                                                        min_df = 5

                                                       )

        )

    ),

    MultinomialNB(fit_prior = False)

)
# wrap the metric in a scorer function

score_func = sk.metrics.make_scorer(sk.metrics.log_loss,

                                    greater_is_better = False,

                                    needs_proba = True)



# run the K-fold cross-validation with K = 5

scores = sk.model_selection.cross_validate(reference,

                                           df,

                                           df.label,

                                           cv = 5,

                                           scoring = score_func,

                                           return_train_score = True)

# output the performance

train_scores = scores['train_score']

test_scores = scores['test_score']

print('log-loss score of your model = %.2f +/- %.2f (training: %.2f +/- %.2f)' % (-np.mean(test_scores), np.std(test_scores, ddof = 1),-np.mean(train_scores), np.std(train_scores, ddof = 1)))
def _penn_to_wordnet(tag):

    '''Converts the corpus tag into a Wordnet tag.'''

    if tag in ("NN", "NNS", "NNP", "NNPS"):

        return nltk.wordnet.NOUN

    if tag in ("JJ", "JJR", "JJS"):

        return nltk.wordnet.wordnet.ADJ

    if tag in ("VB", "VBD", "VBG", "VBN", "VBP", "VBZ"):

        return nltk.wordnet.wordnet.VERB

    if tag in ("RB", "RBR", "RBS"):

        return nltk.wordnet.wordnet.ADV

    return None



def lemmatize(text):

    b = TB(text)

    return ' '.join([Word(w).lemmatize(_penn_to_wordnet(t)) for w,t in b.tags])
df['lemma'] = df.text.transform(lemmatize)
c.fit_transform(df.lemma)

print(len(c.vocabulary_))
csw = sk.feature_extraction.text.CountVectorizer(vocabulary = english_sw)

X = csw.fit_transform(df.text)

sw_df = pd.DataFrame(data = X.T * Y, columns = df.author.unique(), index = csw.get_feature_names())
a = pd.DataFrame(data = X.todense(), columns = csw.get_feature_names(), index = df.index)
a['len'] = df.n_words
d = a.div(df.n_words, axis = 'index') * 100
d['author'] = df.author
sns.pairplot(data = d, vars = ['but', 'then', 'and', 'or'], hue = 'author');
d.columns
b = TB(df.text.iloc[2])

print(b)

b.noun_phrases
b
b.sentiment
df['polarity'] = df.text.transform(lambda x: TB(x).sentiment.polarity)
df['subjectivity'] = df.text.transform(lambda x: TB(x).sentiment.subjectivity)
sns.violinplot(x = 'author', y = 'subjectivity', data = df, inner = 'quartil')