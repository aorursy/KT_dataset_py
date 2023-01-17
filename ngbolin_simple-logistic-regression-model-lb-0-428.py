import pandas as pd

import numpy as np

import nltk

import re

import matplotlib.pyplot as plt

import seaborn as sns

from wordcloud import WordCloud
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
combined = pd.concat([df_train, df_test]).reset_index(drop=True)
combined.head()
print('The training dataset has %d rows and %d columns' % (df_train.shape[0], df_train.shape[1]))

print('The testing dataset has %d rows and %d columns' % (df_test.shape[0], df_test.shape[1]))

print('The combined dataset has %d rows and %d columns' % (combined.shape[0], combined.shape[1]))
np.sum(pd.isnull(combined))
combined.author.value_counts()
import string

import operator

from collections import OrderedDict

sns.set(font_scale=1.25)



def top_20_words(author):

    # Return a cleaned series of lists of words

    common_words_df = (combined[combined['author'] == author].text

                       .apply(lambda x: ''.join([word for word in x if word not in string.punctuation]))

                       .str.lower()

                       .str.split(' '))

    

    # Returns a dictionary where key = words and values = word counts

    dict_of_word_count = {}

    for text in common_words_df:

        for word in text:

            dict_of_word_count[word] = dict_of_word_count.get(word, 0) + 1



    return sorted(dict_of_word_count.items(), key=operator.itemgetter(1), reverse=True)[:20]



def plot_top_20_words(author):

    plt.figure(figsize=(20, 12))

    topwords = top_20_words(author)

    

    words, freq = list(zip(*topwords))[0], list(zip(*topwords))[1]

    

    x_pos = np.arange(len(words)) 

    

    sns.barplot(x_pos, freq)

    plt.xticks(x_pos, words)

    plt.title('Top 20 words of: ' + author)

    plt.show()
plot_top_20_words('EAP')
wordcloud = WordCloud().generate(str(combined[combined.author=='EAP'].text.tolist()))



plt.figure(figsize=(20, 15))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
plot_top_20_words('HPL')
wordcloud = WordCloud().generate(str(combined[combined.author=='HPL'].text.tolist()))



plt.figure(figsize=(20, 15))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
plot_top_20_words('MWS')
wordcloud = WordCloud().generate(str(combined[combined.author=='MWS'].text.tolist()))



plt.figure(figsize=(20, 15))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
combined['sent_length'] = combined.text.apply(lambda x: len(x))
def word_count(text):

    return len(''.join([word.lower() for word in text if word not in string.punctuation]).split(' '))



combined['word_length'] = combined.text.apply(word_count)
combined['punc_marks'] = (combined

                          .text

                          .apply(lambda x: 

                                 len(re.findall('[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]', x))) /

                          combined.sent_length)
combined['cap_letter'] = (combined

                          .text

                          .apply(lambda x: 

                                 len(re.findall('[A-Z]', x))) /

                          combined.sent_length)
combined['avg_word_length'] = combined.sent_length / combined.word_length
plt.figure(figsize=(20, 12))



sns.distplot(combined[combined.author == 'EAP'].sent_length, color = 'salmon', 

             bins=np.linspace(0, 1000, 101), kde=False, norm_hist=True, label = 'EAP')

sns.distplot(combined[combined.author == 'HPL'].sent_length, color = 'steelblue', 

             bins=np.linspace(0, 1000, 101), kde=False, norm_hist=True, label = 'HPL')

sns.distplot(combined[combined.author == 'MWS'].sent_length, color = 'seagreen', 

             bins=np.linspace(0, 1000, 101), kde=False, norm_hist=True, label = 'MWS')



plt.title('Sentence Length')

plt.legend()

plt.show()
plt.figure(figsize=(20, 12))



sns.distplot(combined[combined.author == 'EAP'].word_length, color = 'salmon',

             bins=np.linspace(0, 200, 101), kde=False, norm_hist=True, label = 'EAP')

sns.distplot(combined[combined.author == 'HPL'].word_length, color = 'steelblue', 

             bins=np.linspace(0, 200, 101), kde=False, norm_hist=True, label = 'HPL')

sns.distplot(combined[combined.author == 'MWS'].word_length, color = 'seagreen', 

             bins=np.linspace(0, 200, 101), kde=False, norm_hist=True, label = 'MWS')



plt.title('Word Length')

plt.legend()

plt.show()
plt.figure(figsize=(20, 12))



sns.distplot(combined[combined.author == 'EAP'].punc_marks, color='salmon', label = 'EAP',

             bins = np.linspace(0, 0.2, 101), kde=False, norm_hist=True)

sns.distplot(combined[combined.author == 'HPL'].punc_marks, color='steelblue', label = 'HPL',

             bins = np.linspace(0, 0.2, 101), kde=False, norm_hist=True)

sns.distplot(combined[combined.author == 'MWS'].punc_marks, color='seagreen', label = 'MWS',

             bins = np.linspace(0, 0.2, 101), kde=False, norm_hist=True)

plt.title("Average Number of Punctuation Marks Used")

plt.legend()



plt.show()
plt.figure(figsize=(20, 12))



sns.distplot(combined[combined.author == 'EAP'].cap_letter, color='salmon', label = 'EAP',

             bins = np.linspace(0, 0.2, 101), kde=False, norm_hist=True)

sns.distplot(combined[combined.author == 'HPL'].cap_letter, color='steelblue', label = 'HPL',

             bins = np.linspace(0, 0.2, 101), kde=False, norm_hist=True)

sns.distplot(combined[combined.author == 'MWS'].cap_letter, color='seagreen', label = 'MWS',

             bins = np.linspace(0, 0.2, 101), kde=False, norm_hist=True)

plt.title("Average Number of Capital Letters Used")

plt.xlabel('Average Number of Capital Letters')

plt.legend()



plt.show()
plt.figure(figsize=(20, 12))



sns.distplot(combined[combined.author == 'EAP'].avg_word_length, color='salmon', label = 'EAP',

             bins = np.linspace(0, 12, 61), kde=False, norm_hist=True)

sns.distplot(combined[combined.author == 'HPL'].avg_word_length, color='steelblue', label = 'HPL',

             bins = np.linspace(0, 12, 61), kde=False, norm_hist=True)

sns.distplot(combined[combined.author == 'MWS'].avg_word_length, color='seagreen', label = 'MWS',

             bins = np.linspace(0, 12, 61), kde=False, norm_hist=True)

plt.title("Average Word Length")

plt.xlabel('Average Word Length')

plt.legend()



plt.show()
# Cleaning text - removing punctuation, and converting capital letters to small letters

def list_of_words(text):

    return ''.join([word.lower() for word in text if word not in string.punctuation]).split(' ')



wordlist = combined.text.apply(list_of_words)
# Cleaning text - removing non-alphanumeric characters

def remove_spaces(text):

    return ' '.join([re.sub('[^a-zA-Z0-9]', ' ', word) for word in text]).split(' ')



wordlist = wordlist.apply(remove_spaces)
# Removing stopwords from the list of words - warning: takes a long time 

from nltk.corpus import stopwords



def list_of_nonstopwords(text):

    return [word for word in text if word not in stopwords.words('english')]



nonstopword_list = wordlist.apply(list_of_nonstopwords)
# Lemmatizing the list of non-stop-words

from nltk.stem.wordnet import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()



def lemmatized_words(text):

    return ' '.join([lemmatizer.lemmatize(word) for word in text])



combined['lemmatized_words'] = nonstopword_list.apply(lemmatized_words)
# Stemming the list of non-stop-words

from nltk.stem import PorterStemmer

stemmer = PorterStemmer()



def stemmed_words(text):

    return ' '.join([stemmer.stem(word) for word in text])



combined['stemmed_words'] = nonstopword_list.apply(stemmed_words)
combined.head()
def top_20_words_lemmatized(author):

    # Return a cleaned series of lists of words

    author_stemmed = (combined[combined.author == author].lemmatized_words

                      .apply(lambda text: ''.join([word for word in text])).str.split(' '))



    # Returns a dictionary where key = words and values = word counts

    dict_of_word_count = {}

    for text in author_stemmed:

        for word in text: dict_of_word_count[word] = dict_of_word_count.get(word, 0) + 1



    return sorted(dict_of_word_count.items(), key=operator.itemgetter(1), reverse=True)[:20]



def plot_top_20_words_lemmatized(author):

    plt.figure(figsize=(20, 12))

    topwords = top_20_words_lemmatized(author)

    

    words, freq = list(zip(*topwords))[0], list(zip(*topwords))[1]

    

    x_pos = np.arange(len(words)) 

    sns.barplot(x_pos, freq)

    plt.xticks(x_pos, words)

    plt.title('Top 20 lemmatized_words terms of: ' + author)

    plt.show()
# Plotting top lemmatized words for Edgar Allen Poe

plot_top_20_words_lemmatized('EAP')
wordcloud = WordCloud().generate(str(combined[combined.author=='EAP'].lemmatized_words.tolist()))



plt.figure(figsize=(20, 15))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
# Plotting top lemmatized words for H.P. Lovecraft

plot_top_20_words_lemmatized('HPL')
wordcloud = WordCloud().generate(str(combined[combined.author=='HPL'].lemmatized_words.tolist()))



plt.figure(figsize=(20, 15))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
# Plotting top lemmatized words for Mary Shelley

plot_top_20_words_lemmatized('MWS')
wordcloud = WordCloud().generate(str(combined[combined.author=='MWS'].lemmatized_words.tolist()))



plt.figure(figsize=(20, 15))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
# Conduct POS Tagging (takes a bit of time to run this code)

pos_tags = (combined.text.apply(lambda text: nltk.pos_tag(nltk.word_tokenize(text))))
def pos_tag_count(list_of_postag):

    # Return dictionary of dataframes with postags as keys and counts as values

    dict_of_postags = {}

    for tag in list_of_postag:

        dict_of_postags[tag[1]] = dict_of_postags.get(tag[1], 0) + 1

    return dict_of_postags

        

postags_df = pd.DataFrame(pos_tags.apply(pos_tag_count).to_dict()).T
pos_tag_col = [col for col in postags_df.columns if re.findall('[A-Z]+', col)]



postags_df_ = postags_df[pos_tag_col].fillna(0)
postags_df_['EAP'] = pd.get_dummies(df_train.author).EAP

postags_df_['HPL'] = pd.get_dummies(df_train.author).HPL

postags_df_['MWS'] = pd.get_dummies(df_train.author).MWS



sns.set(font_scale=1)

plt.figure(figsize=(20,12))

sns.heatmap(postags_df_.corr())

plt.show()
del postags_df_['EAP']

del postags_df_['HPL']

del postags_df_['MWS']



combined = pd.merge(combined, postags_df_,

                    left_index=True, right_index=True)
X_train = combined.iloc[:df_train.shape[0]]

X_test = combined.iloc[df_train.shape[0]:]
from sklearn.feature_extraction.text import CountVectorizer



# Use CountVectorizor to remove stop_words, remove tokens that don't appear in at least 3 documents,

# and remove tokens that appear in more than 10% of the documents

vect = CountVectorizer(min_df=3, ngram_range=(1, 10))



train_counts_transformed = vect.fit_transform(X_train.stemmed_words)

test_counts_transformed = vect.transform(X_test.stemmed_words)
train_counts_transformed
from sklearn.feature_extraction.text import TfidfTransformer



tfidf = TfidfTransformer(use_idf=True)



train_tfidf = tfidf.fit_transform(train_counts_transformed)

test_tfidf = tfidf.transform(test_counts_transformed)
X_train_ = pd.merge(X_train, pd.DataFrame(train_tfidf.toarray()),

                    left_index=True, right_index=True)



X_test_ = pd.merge(X_test.reset_index(drop=True), pd.DataFrame(test_tfidf.toarray()), 

                   left_index=True, right_index=True)
import gensim

from sklearn.feature_extraction.text import CountVectorizer



vect = CountVectorizer(min_df=3, ngram_range=(1,5), stop_words='english')



# Fit and transform

text_train = vect.fit_transform(X_train.lemmatized_words)



# Convert sparse matrix to gensim corpus.

corpus = gensim.matutils.Sparse2Corpus(text_train, documents_columns=False)



# Mapping from word IDs to words (To be used in LdaModel's id2word parameter)

id_map = dict((v, k) for k, v in vect.vocabulary_.items())
# Use the gensim.models.ldamodel.LdaModel constructor to estimate 

# LDA model parameters on the corpus, and save to the variable `ldamodel`



random_state = 9410



ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = 6,

                                           id2word = id_map, passes = 6,

                                           random_state = random_state)
ldamodel.show_topics(num_topics=6)
def most_probable_topic(text):

    

    # Transform text into Corpus

    X = vect.transform(text)

    corpus = gensim.matutils.Sparse2Corpus(X, documents_columns=False)

    

    # Return topic distribution

    topic_dist =  ldamodel.inference(corpus)[0]

    

    topics = [max(enumerate(corpus), key=operator.itemgetter(1))[0] for corpus in topic_dist]

    

    return topics
train_topics = pd.get_dummies(most_probable_topic(X_train.lemmatized_words), prefix='topic')

test_topics = pd.get_dummies(most_probable_topic(X_test.lemmatized_words), prefix='topic')
train_topics.head()
X_train = pd.merge(X_train_, train_topics, left_index=True, right_index=True)

X_test = pd.merge(X_test_, test_topics, left_index=True, right_index=True)
print('Training Dimension: ', X_train.shape)

print('Testing Dimension: ', X_test.shape)
plt.figure(figsize=(20,12))



topics = pd.DataFrame(most_probable_topic(X_train.lemmatized_words))



sns.distplot(topics.iloc[combined[combined.author=='EAP'].index.values],

             bins=range(0, 7, 1), kde=False, norm_hist=True, color='steelblue', label='EAP')

sns.distplot(topics.iloc[combined[combined.author=='HPL'].index.values],

             bins=range(0, 7, 1), kde=False, norm_hist=True, color='seagreen', label='HPL')

sns.distplot(topics.iloc[combined[combined.author=='MWS'].index.values],

             bins=range(0, 7, 1), kde=False, norm_hist=True, color='salmon', label='MWS')



plt.title('Topic Distribution')

plt.legend()

plt.show()
from sklearn.model_selection import train_test_split



X_train.columns = [str(feat) for feat in X_train.columns.tolist()]

X_test.columns = [str(feat) for feat in X_test.columns.tolist()]

features = [feat for feat in X_train.columns.tolist() 

            if feat not in ['author', 'id', 'text', 'stemmed_words', 'lemmatized_words']]



X, y = X_train[features], X_train.author.values.ravel()

X_test = X_test[features]



X_subtrain, X_subtest, y_subtrain, y_subtest = train_test_split(X, y, test_size=0.2,

                                                                random_state=random_state)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV



logregr = LogisticRegression(random_state=random_state)



param_grid = {'C': np.logspace(-2, 2, 5)}



clf = GridSearchCV(logregr, param_grid=param_grid, scoring='neg_log_loss', cv=3)

clf.fit(X_subtrain, y_subtrain)
from sklearn.metrics import log_loss



log_loss(y_subtest, clf.predict_proba(X_subtest))
clf.fit(X, y)
X_test_pred = pd.DataFrame(clf.predict_proba(X_test), 

                           columns = ['EAP', 'HPL', 'MWS'])
submission = pd.read_csv('submission.csv')



X_test_pred['id'] = submission['id']

(X_test_pred.set_index('id')

 .reset_index()

 .to_csv('submission_.csv', index=False))