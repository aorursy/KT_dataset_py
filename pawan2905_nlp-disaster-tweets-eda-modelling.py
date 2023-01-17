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
import matplotlib.pyplot as plt

import seaborn as sns

from IPython.display import Markdown, display

def printmd(string):

    display(Markdown(string))

#printmd('**bold**')
import nltk

nltk.download('stopwords')

nltk.download('punkt')

nltk.download('averaged_perceptron_tagger')

nltk.download('wordnet')

from nltk.corpus import stopwords, wordnet

from nltk.stem import WordNetLemmatizer

from nltk.tokenize import word_tokenize

from nltk.probability import FreqDist



#Count vectorizer for N grams

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer



# Nltk for tekenize and stopwords

from nltk.corpus import stopwords 

from nltk.tokenize import word_tokenize 

from wordcloud import WordCloud, STOPWORDS

from collections import Counter, defaultdict





# For customizing our plots.



from matplotlib.ticker import MaxNLocator

import matplotlib.gridspec as gridspec

import matplotlib.patches as mpatches

train_df = pd.read_csv('../input/nlp-getting-started/train.csv')

test_df= pd.read_csv("../input/nlp-getting-started/test.csv")

sub_df = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")
display(train_df.sample(5))

display(test_df.sample(5))
print("Shape of the training data...",train_df.shape)

print("Shape of the testing data...",test_df.shape)
import re

import string
def remove_URL(text):

    url =re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'',text)



def remove_emoji(text):

    emoji_pattern = re.compile(

        '['

        u'\U0001F600-\U0001F64F'  # emoticons

        u'\U0001F300-\U0001F5FF'  # symbols & pictographs

        u'\U0001F680-\U0001F6FF'  # transport & map symbols

        u'\U0001F1E0-\U0001F1FF'  # flags (iOS)

        u'\U00002702-\U000027B0'

        u'\U000024C2-\U0001F251'

        ']+',

        flags=re.UNICODE)

    return emoji_pattern.sub(r'', text)





def remove_html(text):

    html = re.compile(r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')

    return re.sub(html, '', text)





def remove_punct(text):

    table = str.maketrans('', '', string.punctuation)

    return text.translate(table)
train_df["text_clean"] = train_df["text"].apply(lambda x:remove_URL(x))

train_df["text_clean"] = train_df["text_clean"].apply(lambda x : remove_emoji(x))

train_df["text_clean"] = train_df["text_clean"].apply(lambda x : remove_html(x))

train_df["text_clean"] = train_df["text_clean"].apply(lambda x : remove_punct(x))
train_df["tokenized"] = train_df["text_clean"].apply(word_tokenize)
train_df.head()
train_df['lower'] = train_df['tokenized'].apply(lambda x:[word.lower() for word in x])

train_df.head()
from wordcloud import WordCloud, STOPWORDS

stop = set(stopwords.words('english'))
train_df['stopwords_removed'] = train_df['lower'].apply(lambda x: [word for word in x if word not in stop])

train_df.head()




train_df['pos_tags'] = train_df['stopwords_removed'].apply(nltk.tag.pos_tag)



train_df.head()


def get_wordnet_pos(tag):

    if tag.startswith('J'):

        return wordnet.ADJ

    elif tag.startswith('V'):

        return wordnet.VERB

    elif tag.startswith('N'):

        return wordnet.NOUN

    elif tag.startswith('R'):

        return wordnet.ADV

    else:

        return wordnet.NOUN





train_df['wordnet_pos'] = train_df['pos_tags'].apply(

    lambda x: [(word, get_wordnet_pos(pos_tag)) for (word, pos_tag) in x])



train_df.head()
# Applying word lemmatizer.



wnl = WordNetLemmatizer()



train_df['lemmatized'] = train_df['wordnet_pos'].apply(

    lambda x: [wnl.lemmatize(word, tag) for word, tag in x])



train_df['lemmatized'] = train_df['lemmatized'].apply(

    lambda x: [word for word in x if word not in stop])



train_df['lemma_str'] = [' '.join(map(str, l)) for l in train_df['lemmatized']]

train_df

train_df.head()
fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(18, 6), dpi=100)

sns.countplot(train_df['target'], ax=axes[0])

axes[1].pie(train_df['target'].value_counts(),

            labels=['Not Disaster', 'Disaster'],

            autopct='%1.2f%%',

            shadow=True,

            explode=(0.05, 0),

            startangle=60)

fig.suptitle('Distribution of the Tweets', fontsize=24)

plt.show()
# Creating a new feature for the visualization.



train_df['Character Count'] = train_df['text_clean'].apply(lambda x: len(str(x)))





def plot_dist3(df, feature, title):

    # Creating a customized chart. and giving in figsize and everything.

    fig = plt.figure(constrained_layout=True, figsize=(18, 8))

    # Creating a grid of 3 cols and 3 rows.

    grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)



    # Customizing the histogram grid.

    ax1 = fig.add_subplot(grid[0, :2])

    # Set the title.

    ax1.set_title('Histogram')

    # plot the histogram.

    sns.distplot(df.loc[:, feature],

                 hist=True,

                 kde=True,

                 ax=ax1,

                 color='#e74c3c')

    ax1.set(ylabel='Frequency')

    ax1.xaxis.set_major_locator(MaxNLocator(nbins=20))



    # Customizing the ecdf_plot.

    ax2 = fig.add_subplot(grid[1, :2])

    # Set the title.

    ax2.set_title('Empirical CDF')

    # Plotting the ecdf_Plot.

    sns.distplot(df.loc[:, feature],

                 ax=ax2,

                 kde_kws={'cumulative': True},

                 hist_kws={'cumulative': True},

                 color='#e74c3c')

    ax2.xaxis.set_major_locator(MaxNLocator(nbins=20))

    ax2.set(ylabel='Cumulative Probability')



    # Customizing the Box Plot.

    ax3 = fig.add_subplot(grid[:, 2])

    # Set title.

    ax3.set_title('Box Plot')

    # Plotting the box plot.

    sns.boxplot(x=feature, data=df, orient='v', ax=ax3, color='#e74c3c')

    ax3.yaxis.set_major_locator(MaxNLocator(nbins=25))



    plt.suptitle(f'{title}', fontsize=24)
plot_dist3(train_df[train_df['target'] == 0], 'Character Count',

           'Characters Per "Non Disaster" Tweet')
plot_dist3(train_df[train_df['target'] == 1], 'Character Count',

           'Characters Per "Disaster" Tweet')
def plot_word_number_histogram(textno, textye):

    

    """A function for comparing word counts"""



    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(18, 6), sharey=True)

    sns.distplot(textno.str.split().map(lambda x: len(x)), ax=axes[0], color='#e74c3c')

    sns.distplot(textye.str.split().map(lambda x: len(x)), ax=axes[1], color='#e74c3c')

    

    axes[0].set_xlabel('Word Count')

    axes[0].set_ylabel('Frequency')

    axes[0].set_title('Non Disaster Tweets')

    axes[1].set_xlabel('Word Count')

    axes[1].set_title('Disaster Tweets')

    

    fig.suptitle('Words Per Tweet', fontsize=24, va='baseline')

    

    fig.tight_layout()
plot_word_number_histogram(train_df[train_df['target'] == 0]['text'],

                           train_df[train_df['target'] == 1]['text'])
def plot_word_len_histogram(textno, textye):

    

    """A function for comparing average word length"""

    

    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(18, 6), sharey=True)

    sns.distplot(textno.str.split().apply(lambda x: [len(i) for i in x]).map(

        lambda x: np.mean(x)),

                 ax=axes[0], color='#e74c3c')

    sns.distplot(textye.str.split().apply(lambda x: [len(i) for i in x]).map(

        lambda x: np.mean(x)),

                 ax=axes[1], color='#e74c3c')

    

    axes[0].set_xlabel('Word Length')

    axes[0].set_ylabel('Frequency')

    axes[0].set_title('Non Disaster Tweets')

    axes[1].set_xlabel('Word Length')

    axes[1].set_title('Disaster Tweets')

    

    fig.suptitle('Mean Word Lengths', fontsize=24, va='baseline')

    fig.tight_layout()
plot_word_len_histogram(train_df[train_df['target'] == 0]['text'],

                        train_df[train_df['target'] == 1]['text'])
lis = [

    train_df[train_df['target'] == 0]['lemma_str'],

    train_df[train_df['target'] == 1]['lemma_str']

]
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

axes = axes.flatten()



for i, j in zip(lis, axes):

    try:

        new = i.str.split()

        new = new.values.tolist()

        corpus = [word.lower() for i in new for word in i]

        dic = defaultdict(int)

        for word in corpus:

            if word in stop:

                dic[word] += 1



        top = sorted(dic.items(), key=lambda x: x[1], reverse=True)[:15]

        x, y = zip(*top)

        df = pd.DataFrame([x, y]).T

        df = df.rename(columns={0: 'Stopword', 1: 'Count'})

        sns.barplot(x='Count', y='Stopword', data=df, palette='plasma', ax=j)

        plt.tight_layout()

    except:

        plt.close()

        print('No stopwords left in texts.')

        break
# Displaying most common words.



fig, axes = plt.subplots(1, 2, figsize=(18, 8))

axes = axes.flatten()



for i, j in zip(lis, axes):



    new = i.str.split()

    new = new.values.tolist()

    corpus = [word for i in new for word in i]



    counter = Counter(corpus)

    most = counter.most_common()

    x, y = [], []

    for word, count in most[:30]:

        if (word not in stop):

            x.append(word)

            y.append(count)



    sns.barplot(x=y, y=x, palette='plasma', ax=j)

axes[0].set_title('Non Disaster Tweets')



axes[1].set_title('Disaster Tweets')

axes[0].set_xlabel('Count')

axes[0].set_ylabel('Word')

axes[1].set_xlabel('Count')

axes[1].set_ylabel('Word')



fig.suptitle('Most Common Unigrams', fontsize=24, va='baseline')

plt.tight_layout()
# !pip install -U spacy
# !python -m spacy download en_core_web_lg

# !python -m spacy download en_core_web_sm

import spacy

nlp = spacy.load('en_core_web_sm')
import spacy

nlp = spacy.load("en_core_web_sm")



doc = nlp("Company Y is planning to acquire stake in X company for $23 billion")

for token in doc:

    print(token.text, token.pos_, token.dep_)
import spacy

nlp = spacy.load('en_core_web_sm')



# Create an nlp object

doc = nlp("He went to play cricket with friends in the stadium")
nlp.pipe_names
nlp.disable_pipes('tagger', 'parser')
#Let’s again check the active pipeline component:



nlp.pipe_names
import spacy



nlp = spacy.load("en_core_web_sm")

doc = nlp("Reliance is looking at buying U.K. based analytics startup for $7.2 billion")

for token in doc:

    print(token.text)
import spacy 

nlp = spacy.load('en_core_web_sm')



# Create an nlp object

doc = nlp("Reliance is looking at buying U.K. based analytics startup for $7 billion")

 

# Iterate over the tokens

for token in doc:

    # Print the token and its part-of-speech tag

    print(token, token.tag_, token.pos_, spacy.explain(token.tag_))
import spacy

from spacy import displacy



doc = nlp("Reliance is looking at buying U.K. based analytics startup for $7 billion")

displacy.render(doc, style="dep" , jupyter=True)
import spacy 

nlp = spacy.load('en_core_web_sm')



# Create an nlp object

doc = nlp("Reliance is looking at buying U.K. based analytics startup for $7 billion")

 

# Iterate over the tokens

for token in doc:

    # Print the token and its part-of-speech tag

    print(token.text, "-->", token.dep_)
spacy.explain("nsubj"), spacy.explain("ROOT"), spacy.explain("aux"), spacy.explain("advcl"), spacy.explain("dobj")
import spacy 

nlp = spacy.load('en_core_web_sm')



# Create an nlp object

doc = nlp("Reliance is looking at buying U.K. based analytics startup for $7 billion")

 

# Iterate over the tokens

for token in doc:

    # Print the token and its part-of-speech tag

    print(token.text, "-->", token.lemma_)
import spacy 

nlp = spacy.load('en_core_web_sm')



# Create an nlp object

doc = nlp("Reliance is looking at buying U.K. based analytics startup for $7 billion.This is India.India is great")

 

sentences = list(doc.sents)

len(sentences)

for sentence in sentences:

     print (sentence)
import spacy



nlp = spacy.load("en_core_web_sm")

doc = nlp("Reliance is looking at buying U.K. based analytics startup for $7 billion")



for ent in doc.ents:

    print(ent.text, ent.start_char, ent.end_char, ent.label_)
import spacy

from spacy import displacy

nlp = spacy.load("en_core_web_sm")

doc= nlp(u"""The Amazon rainforest,[a] alternatively, the Amazon Jungle, also known in English as Amazonia, is a moist broadleaf tropical rainforest in the Amazon biome that covers most of the Amazon basin of South America. This basin encompasses 7,000,000 km2 (2,700,000 sq mi), of which 5,500,000 km2 (2,100,000 sq mi) are covered by the rainforest. This region includes territory belonging to nine nations.



The majority of the forest is contained within Brazil, with 60% of the rainforest, followed by Peru with 13%, Colombia with 10%, and with minor amounts in Bolivia, Ecuador, French Guiana, Guyana, Suriname, and Venezuela. Four nations have "Amazonas" as the name of one of their first-level administrative regions and France uses the name "Guiana Amazonian Park" for its rainforest protected area. The Amazon represents over half of the planet's remaining rainforests,[2] and comprises the largest and most biodiverse tract of tropical rainforest in the world, with an estimated 390 billion individual trees divided into 16,000 species.[3]



Etymology

The name Amazon is said to arise from a war Francisco de Orellana fought with the Tapuyas and other tribes. The women of the tribe fought alongside the men, as was their custom.[4] Orellana derived the name Amazonas from the Amazons of Greek mythology, described by Herodotus and Diodorus.[4]



History

See also: History of South America § Amazon, and Amazon River § History

Tribal societies are well capable of escalation to all-out wars between tribes. Thus, in the Amazonas, there was perpetual animosity between the neighboring tribes of the Jivaro. Several tribes of the Jivaroan group, including the Shuar, practised headhunting for trophies and headshrinking.[5] The accounts of missionaries to the area in the borderlands between Brazil and Venezuela have recounted constant infighting in the Yanomami tribes. More than a third of the Yanomamo males, on average, died from warfare.[6]""")



entities=[(i, i.label_, i.label) for i in doc.ents]

entities
displacy.render(doc, style = "ent",jupyter = True)
import spacy



nlp = spacy.load("en_core_web_lg")

tokens = nlp("dog cat banana afskfsd")



for token in tokens:

    print(token.text, token.has_vector, token.vector_norm, token.is_oov)
import spacy



nlp = spacy.load("en_core_web_lg")  # make sure to use larger model!

tokens = nlp("dog cat banana")



for token1 in tokens:

    for token2 in tokens:

        print(token1.text, token2.text, token1.similarity(token2))
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

from sklearn.base import TransformerMixin

from sklearn.pipeline import Pipeline
import string

from spacy.lang.en.stop_words import STOP_WORDS

from spacy.lang.en import English



# Create our list of punctuation marks

punctuations = string.punctuation



# Create our list of stopwords

nlp = spacy.load('en')

stop_words = spacy.lang.en.stop_words.STOP_WORDS



# Load English tokenizer, tagger, parser, NER and word vectors

parser = English()



# Creating our tokenizer function

def spacy_tokenizer(sentence):

    # Creating our token object, which is used to create documents with linguistic annotations.

    mytokens = parser(sentence)



    # Lemmatizing each token and converting each token into lowercase

    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]



    # Removing stop words

    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]



    # return preprocessed list of tokens

    return mytokens
# Custom transformer using spaCy

class predictors(TransformerMixin):

    def transform(self, X, **transform_params):

        # Cleaning Text

        return [clean_text(text) for text in X]



    def fit(self, X, y=None, **fit_params):

        return self



    def get_params(self, deep=True):

        return {}



# Basic function to clean the text

def clean_text(text):

    # Removing spaces and converting text into lowercase

    return text.strip().lower()
bow_vector = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1))

tfidf_vector = TfidfVectorizer(tokenizer = spacy_tokenizer)

train_df.head()
from sklearn.model_selection import train_test_split



X = train_df['text'] # the features we want to analyze

ylabels = train_df['target'] # the labels, or answers, we want to test against



X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.3)
# Logistic Regression Classifier

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()



# Create pipeline using Bag of Words

pipe = Pipeline([("cleaner", predictors()),

                 ('vectorizer', bow_vector),

                 ('classifier', classifier)])



# model generation

pipe.fit(X_train,y_train)
from sklearn import metrics

# Predicting with a test dataset

predicted = pipe.predict(X_test)



# Model Accuracy

print("Logistic Regression Accuracy:",metrics.accuracy_score(y_test, predicted))

print("Logistic Regression Precision:",metrics.precision_score(y_test, predicted))

print("Logistic Regression Recall:",metrics.recall_score(y_test, predicted))
# Applying a first round of text cleaning techniques



def clean_text(text):

    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation

    and remove words containing numbers.'''

    text = text.lower()

    text = re.sub('\[.*?\]', '', text)

    text = re.sub('https?://\S+|www\.\S+', '', text)

    text = re.sub('<.*?>+', '', text)

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub('\n', '', text)

    text = re.sub('\w*\d\w*', '', text)

    return text



# Applying the cleaning function to both test and training datasets

train_df['text'] = train_df['text'].apply(lambda x: clean_text(x))

test_df['text'] = test_df['text'].apply(lambda x: clean_text(x))



# Let's take a look at the updated text

train_df['text'].head()
# A disaster tweet

disaster_tweets = train_df[train_df['target']==1]['text']

disaster_tweets.values[1]

'Forest fire near La Ronge Sask. Canada'

#not a disaster tweet

non_disaster_tweets = train_df[train_df['target']==0]['text']

non_disaster_tweets.values[1]
from wordcloud import WordCloud

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[26, 8])

wordcloud1 = WordCloud( background_color='white',

                        width=600,

                        height=400).generate(" ".join(disaster_tweets))

ax1.imshow(wordcloud1)

ax1.axis('off')

ax1.set_title('Disaster Tweets',fontsize=40);



wordcloud2 = WordCloud( background_color='white',

                        width=600,

                        height=400).generate(" ".join(non_disaster_tweets))

ax2.imshow(wordcloud2)

ax2.axis('off')

ax2.set_title('Non Disaster Tweets',fontsize=40);
 

text = "Are you coming , aren't you"

tokenizer1 = nltk.tokenize.WhitespaceTokenizer()

tokenizer2 = nltk.tokenize.TreebankWordTokenizer()

tokenizer3 = nltk.tokenize.WordPunctTokenizer()

tokenizer4 = nltk.tokenize.RegexpTokenizer(r'\w+')



print("Example Text: ",text)

print("------------------------------------------------------------------------------------------------")

print("Tokenization by whitespace:- ",tokenizer1.tokenize(text))

print("Tokenization by words using Treebank Word Tokenizer:- ",tokenizer2.tokenize(text))

print("Tokenization by punctuation:- ",tokenizer3.tokenize(text))

print("Tokenization by regular expression:- ",tokenizer4.tokenize(text))
# Tokenizing the training and the test set

tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

train_df['text'] = train_df['text'].apply(lambda x: tokenizer.tokenize(x))

test_df['text'] = test_df['text'].apply(lambda x: tokenizer.tokenize(x))

train_df['text'].head()
 

def remove_stopwords(text):

    """

    Removing stopwords belonging to english language

    

    """

    words = [w for w in text if w not in stopwords.words('english')]

    return words





train_df['text'] = train_df['text'].apply(lambda x : remove_stopwords(x))

test_df['text'] = test_df['text'].apply(lambda x : remove_stopwords(x))

train_df.head()
# Stemming and Lemmatization examples

text = "feet cats wolves talked"



tokenizer = nltk.tokenize.TreebankWordTokenizer()

tokens = tokenizer.tokenize(text)



# Stemmer

stemmer = nltk.stem.PorterStemmer()

print("Stemming the sentence: ", " ".join(stemmer.stem(token) for token in tokens))



# Lemmatizer

lemmatizer=nltk.stem.WordNetLemmatizer()

print("Lemmatizing the sentence: ", " ".join(lemmatizer.lemmatize(token) for token in tokens))
# After preprocessing, the text format

def combine_text(list_of_text):

    '''Takes a list of text and combines them into one large chunk of text.'''

    combined_text = ' '.join(list_of_text)

    return combined_text



train_df['text'] = train_df['text'].apply(lambda x : combine_text(x))

test_df['text'] = test_df['text'].apply(lambda x : combine_text(x))

train_df['text']

train_df.head()


# text preprocessing function

def text_preprocessing(text):

    """

    Cleaning and parsing the text.



    """

    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

    

    nopunc = clean_text(text)

    tokenized_text = tokenizer.tokenize(nopunc)

    remove_stopwords = [w for w in tokenized_text if w not in stopwords.words('english')]

    combined_text = ' '.join(remove_stopwords)

    return combined_text
count_vectorizer = CountVectorizer()

train_vectors = count_vectorizer.fit_transform(train_df['text'])

test_vectors = count_vectorizer.transform(test_df["text"])



## Keeping only non-zero elements to preserve space 

print(train_vectors[0].todense())
tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2))

train_tfidf = tfidf.fit_transform(train_df['text'])

test_tfidf = tfidf.transform(test_df["text"])
from sklearn import model_selection


# Fitting a simple Logistic Regression on Counts

clf = LogisticRegression(C=1.0)

scores = model_selection.cross_val_score(clf, train_vectors, train_df["target"], cv=5, scoring="f1")

scores
clf.fit(train_vectors, train_df["target"])
# Fitting a simple Logistic Regression on TFIDF

clf_tfidf = LogisticRegression(C=1.0)

scores = model_selection.cross_val_score(clf_tfidf, train_tfidf, train_df["target"], cv=5, scoring="f1")

scores
#Naives Bayes Classifier

#Well, this is a decent score. Let's try with another model that is said to work well with text data : Naive Bayes.



# Fitting a simple Naive Bayes on Counts

from sklearn.naive_bayes import MultinomialNB



clf_NB = MultinomialNB()

scores = model_selection.cross_val_score(clf_NB, train_vectors, train_df["target"], cv=5, scoring="f1")

scores
clf_NB.fit(train_vectors, train_df["target"])
# Fitting a simple Naive Bayes on TFIDF

clf_NB_TFIDF = MultinomialNB()

scores = model_selection.cross_val_score(clf_NB_TFIDF, train_tfidf, train_df["target"], cv=5, scoring="f1")

scores
clf_NB_TFIDF.fit(train_tfidf, train_df["target"])
##XGBoost

import xgboost as xgb

clf_xgb = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, 

                        subsample=0.8, nthread=10, learning_rate=0.1)

scores = model_selection.cross_val_score(clf_xgb, train_vectors, train_df["target"], cv=5, scoring="f1")

scores
import xgboost as xgb

clf_xgb_TFIDF = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, 

                        subsample=0.8, nthread=10, learning_rate=0.1)

scores = model_selection.cross_val_score(clf_xgb_TFIDF, train_tfidf, train_df["target"], cv=5, scoring="f1")

scores
def submission(submission_file_path,model,test_vectors):

    sample_submission = pd.read_csv(submission_file_path)

    sample_submission["target"] = model.predict(test_vectors)

    sample_submission.to_csv("submissionxgb.csv", index=False)

submission_file_path = "../input/nlp-getting-started/sample_submission.csv"

test_vectors=test_tfidf

submission(submission_file_path,clf_NB_TFIDF,test_vectors)
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from nltk.corpus import stopwords

from nltk.util import ngrams

from sklearn.feature_extraction.text import CountVectorizer

from collections import defaultdict

from collections import  Counter

plt.style.use('ggplot')

stop=set(stopwords.words('english'))

import re

from nltk.tokenize import word_tokenize

import gensim

import string

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from tqdm import tqdm

from keras.models import Sequential

from keras.layers import Embedding,LSTM,Dense,SpatialDropout1D

from keras.initializers import Constant

from sklearn.model_selection import train_test_split

from keras.optimizers import Adam
tweet= pd.read_csv('../input/nlp-getting-started/train.csv')

test=pd.read_csv('../input/nlp-getting-started/test.csv')

tweet.head(3)
df=pd.concat([tweet,test])

df.shape
## Removing urls

example="New competition launched :https://www.kaggle.com/c/nlp-getting-started"

def remove_URL(text):

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'',text)



remove_URL(example)
df['text']=df['text'].apply(lambda x : remove_URL(x))
#Removing HTML tags

example = """<div>

<h1>Real or Fake</h1>

<p>Kaggle </p>

<a href="https://www.kaggle.com/c/nlp-getting-started">getting started</a>

</div>"""
def remove_html(text):

    html=re.compile(r'<.*?>')

    return html.sub(r'',text)

print(remove_html(example))
df['text']=df['text'].apply(lambda x : remove_html(x))
# Reference : https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b

def remove_emoji(text):

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', text)



remove_emoji("Omg another Earthquake 😔😔")
df['text']=df['text'].apply(lambda x: remove_emoji(x))

## Removing punctuations

def remove_punct(text):

    table=str.maketrans('','',string.punctuation)

    return text.translate(table)



example="I am a #king"

print(remove_punct(example))
df['text']=df['text'].apply(lambda x : remove_punct(x))


!pip install pyspellchecker
from spellchecker import SpellChecker



spell = SpellChecker()

def correct_spellings(text):

    corrected_text = []

    misspelled_words = spell.unknown(text.split())

    for word in text.split():

        if word in misspelled_words:

            corrected_text.append(spell.correction(word))

        else:

            corrected_text.append(word)

    return " ".join(corrected_text)

        

text = "corect me plese"

correct_spellings(text)


def create_corpus(df):

    corpus=[]

    for tweet in tqdm(df['text']):

        words=[word.lower() for word in word_tokenize(tweet) if((word.isalpha()==1) & (word not in stop))]

        corpus.append(words)

    return corpus

        
corpus=create_corpus(df)
embedding_dict={}

with open('../input/glove-global-vectors-for-word-representation/glove.6B.100d.txt','r') as f:

    for line in f:

        values=line.split()

        word=values[0]

        vectors=np.asarray(values[1:],'float32')

        embedding_dict[word]=vectors

f.close()
MAX_LEN=50

tokenizer_obj=Tokenizer()

tokenizer_obj.fit_on_texts(corpus)

sequences=tokenizer_obj.texts_to_sequences(corpus)



tweet_pad=pad_sequences(sequences,maxlen=MAX_LEN,truncating='post',padding='post')
word_index=tokenizer_obj.word_index

print('Number of unique words:',len(word_index))
num_words=len(word_index)+1

embedding_matrix=np.zeros((num_words,100))



for word,i in tqdm(word_index.items()):

    if i > num_words:

        continue

    

    emb_vec=embedding_dict.get(word)

    if emb_vec is not None:

        embedding_matrix[i]=emb_vec


model=Sequential()



embedding=Embedding(num_words,100,embeddings_initializer=Constant(embedding_matrix),

                   input_length=MAX_LEN,trainable=False)



model.add(embedding)

model.add(SpatialDropout1D(0.2))

model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(1, activation='sigmoid'))





optimzer=Adam(learning_rate=1e-5)



model.compile(loss='binary_crossentropy',optimizer=optimzer,metrics=['accuracy'])
model.summary()
train=tweet_pad[:tweet.shape[0]]

test=tweet_pad[tweet.shape[0]:]
X_train,X_test,y_train,y_test=train_test_split(train,tweet['target'].values,test_size=0.15)

print('Shape of train',X_train.shape)

print("Shape of Validation ",X_test.shape)
history=model.fit(X_train,y_train,batch_size=4,epochs=15,validation_data=(X_test,y_test),verbose=2)
#Making our submission

sample_sub=pd.read_csv('../input/nlp-getting-started/sample_submission.csv')
y_pre=model.predict(test)

y_pre=np.round(y_pre).astype(int).reshape(3263)

sub=pd.DataFrame({'id':sample_sub['id'].values.tolist(),'target':y_pre})

sub.to_csv('submission.csv',index=False)
sub.head()