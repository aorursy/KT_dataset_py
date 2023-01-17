import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import nltk

import warnings



warnings.simplefilter("ignore")



print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#ques = pd.read_csv('../input/stacksample/Questions.csv', encoding='ISO-8859-1')

ques = pd.read_csv('../input/Questions.csv', encoding='ISO-8859-1')

tags = pd.read_csv('../input/Tags.csv', encoding='ISO-8859-1')

ans = pd.read_csv('../input/Answers.csv', encoding='ISO-8859-1')
ques.head()
ans.head()
tags.head()
print('Ques shape: ', ques.shape)

print('Ans shape: ', ans.shape)

print('Tags shape: ', tags.shape)
import collections

import math

ans_per_question = collections.Counter(ans['ParentId'])

#{quesId: ansCount}

# { 406760: 408, 100420: 100, 40480: 69, 490420: 67, 226970: 55, 202750: 51, 17054000: 49 }

# ans_per_question.most_common() =>> [(406760, 408), (38210, 316), (23930, 129), (100420, 100)]

quesId,nosAnswers = zip(*ans_per_question.most_common())

import matplotlib.pyplot as plt

N=20

plt.bar(range(N), nosAnswers[:N], align='center', alpha=0.5)

plt.ylabel('Number of Answers per Question')

plt.xlabel('Question Id')

plt.title('Distribution of Answers per question')

plt.text(3,400,"Avegrage answers per question "+str(math.ceil((np.mean(nosAnswers)))))



plt.show()
ans_freq_counter = collections.Counter(ans_per_question.values())



ans_count,nosQuestions = zip(*ans_freq_counter.most_common())

N=10

plt.bar(ans_count[:N], nosQuestions[:N], align='center', alpha=0.5)

plt.ylabel('Number of Questions')

plt.xlabel('Answer count')

plt.title('Questions vs Their Answer count')

plt.text(5,500000,"Avegrage answers per question "+str(math.ceil((np.mean(nosAnswers)))))



plt.show()
tags_per_question = collections.Counter(tags['Id'])

# { 406760: 8, 100420: 7, 40480: 6, 490420: 5, 226970: 5, 202750: 4, 17054000: 2 }

# ans_per_question.most_common() =>> [(406760, 8), (38210, 7), (23930, 6), (100420, 2)]

tags_freq_counter = collections.Counter(tags_per_question.values())



tags_count,nosQuestions = zip(*tags_freq_counter.most_common())

N=10

plt.bar(tags_count[:N], nosQuestions[:N], align='center', alpha=0.5)

plt.ylabel('Number of Questions')

plt.xlabel('Tags count')

plt.title('Questions vs Their tags count')

plt.text(2,340000,"Avegrage Tags per question "+str(math.ceil((np.mean(tags_count)))))



plt.show()
print('Popular tags')

tagCount =  collections.Counter(list(tags['Tag']))

tagName,freq = zip(*tagCount.most_common(15))

plt.bar(tagName, freq )

plt.xticks(rotation='vertical')

plt.ylabel('Tag Count')

plt.xlabel('Tags name')

plt.title('Tags vs tags count')

plt.show()
import datetime

ques['datetime'] = pd.to_datetime(ques['CreationDate'])

ques.set_index('datetime', inplace=True)
weeklyQues = ques.resample('M').count()

weeklyQues.head()
weeklyQues['datetime'] = weeklyQues.index

weeklyQues.plot(x='datetime', y='Title', kind='line', lw=0.75, c='r')
#tags.info()

tags['Tag'] = tags['Tag'].astype(str)

grouped_tags = tags.groupby("Id")['Tag'].apply(lambda tags: ' '.join(tags))
grouped_tags_final = pd.DataFrame({'Id':grouped_tags.index, 'Tags':grouped_tags})

grouped_tags_final.head()

grouped_tags.reset_index()

ques.drop(columns=['OwnerUserId', 'CreationDate', 'ClosedDate'], inplace=True)
score_gt_5 = ques['Score'] >= 5

ques = ques[score_gt_5]

ques.head()
ques.isnull().sum()
ques.duplicated().sum()
merged_ques = ques.merge(grouped_tags_final, on='Id')
merged_ques.drop(columns=['Id', 'Score'], inplace=True)
merged_ques.head(2)
merged_ques['Tags'] = merged_ques['Tags'].apply(lambda x: x.split())
flat_list = [item for sublist in merged_ques['Tags'].values for item in sublist]
keywords = nltk.FreqDist(flat_list)



keywords = nltk.FreqDist(keywords)



frequencies_words = keywords.most_common(100)

tags_features = [word[0] for word in frequencies_words]
def most_common(tags):

    tags_filtered = []

    for i in range(0, len(tags)):

        if tags[i] in tags_features:

            tags_filtered.append(tags[i])

    return tags_filtered
merged_ques['Tags'] = merged_ques['Tags'].apply(lambda x: most_common(x))

merged_ques['Tags'] = merged_ques['Tags'].apply(lambda x: x if len(x)>0 else None)
merged_ques.dropna(subset=['Tags'], inplace=True)
merged_ques.shape
merged_ques.isnull().sum()
merged_ques.head()


from nltk.tokenize.toktok import ToktokTokenizer

from nltk.stem.wordnet import WordNetLemmatizer

from bs4 import BeautifulSoup

from nltk.corpus import wordnet

from nltk.corpus import stopwords

from string import punctuation

import unicodedata

import spacy

import nltk

import re



nlp = spacy.load('en_core_web_sm', parse=True, tag=True, entity=True)

stopword_list = stopwords.words('english')



tokenizer = ToktokTokenizer()

lemmatizer = WordNetLemmatizer()

lemma=WordNetLemmatizer()

def strip_html_tags(text):

    return BeautifulSoup(text, "html.parser").get_text()
def remove_accented_chars(text):

    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    return text

def remove_special_characters(text):

    pattern = r'[^a-zA-z0-9#\s]'

    text = re.sub(pattern, '', text)

    return text

def simple_stemmer(text):

    ps = nltk.porter.PorterStemmer()

    text = ' '.join([ps.stem(word) for word in text.split()])

    return text
def nltk2wn_tag(nltk_tag):

  if nltk_tag.startswith('J'):

    return wordnet.ADJ

  elif nltk_tag.startswith('V'):

    return wordnet.VERB

  elif nltk_tag.startswith('N'):

    return wordnet.NOUN

  elif nltk_tag.startswith('R'):

    return wordnet.ADV

  else:          

    return None
def lemmatize_text(sentence):

  nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))

  wn_tagged = map(lambda x: (x[0], nltk2wn_tag(x[1])), nltk_tagged)

  res_words = []

  for word, tag in wn_tagged:

    if tag is None:            

      res_words.append(word)

    else:

      res_words.append(lemmatizer.lemmatize(word, tag))

  return " ".join(res_words)
def expand_contractions(text):

    text = text.lower()

    text = re.sub(r"ain't", "is not ", text)

    text = re.sub(r"aren't", "are not ", text)

    text = re.sub(r"can't", "cannot ", text)

    text = re.sub(r"can't've", "cannot have ", text)

    text = re.sub(r"'cause", "because ", text)

    text = re.sub(r"could've", "could have ", text)

    text = re.sub(r"couldn't", "could not ", text)

    text = re.sub(r"couldn't've", "could not have ", text)

    text = re.sub(r"didn't", "did not ", text)

    text = re.sub(r"doesn't", "does not ", text)

    text = re.sub(r"don't", "do not ", text)

    text = re.sub(r"hadn't", "had not ", text)

    text = re.sub(r"hadn't've", "had not have ", text)

    text = re.sub(r"hasn't", "has not ", text)

    text = re.sub(r"haven't", "have not ", text)

    text = re.sub(r"he'd", "he would ", text)

    text = re.sub(r"he'd've", "he would have ", text)

    text = re.sub(r"he'll", "he will ", text)

    text = re.sub(r"he'll've", "he he will have ", text)

    text = re.sub(r"he's", "he is ", text)

    text = re.sub(r"how'd", "how did ", text)

    text = re.sub(r"how'd'y", "how do you ", text)

    text = re.sub(r"how'll", "how will ", text)

    text = re.sub(r"how's", "how is ", text)

    text = re.sub(r"I'd", "I would ", text)

    text = re.sub(r"I'd've", "I would have ", text)

    text = re.sub(r"I'll", "I will ", text)

    text = re.sub(r"I'll've", "I will have ", text)

    text = re.sub(r"I'm", "I am ", text)

    text = re.sub(r"I've", "I have ", text)

    text = re.sub(r"i'd", "i would ", text)

    text = re.sub(r"i'd've", "i would have ", text)

    text = re.sub(r"i'll", "i will ", text)

    text = re.sub(r"i'll've", "i will have ", text)

    text = re.sub(r"i'm", "i am ", text)

    text = re.sub(r"i've", "i have ", text)

    text = re.sub(r"isn't", "is not ", text)

    text = re.sub(r"it'd", "it would ", text)

    text = re.sub(r"it'd've", "it would have ", text)

    text = re.sub(r"it'll", "it will ", text)

    text = re.sub(r"it'll've", "it will have ", text)

    text = re.sub(r"it's", "it is ", text)

    text = re.sub(r"let's", "let us ", text)

    text = re.sub(r"ma'am", "madam ", text)

    text = re.sub(r"mayn't", "may not ", text)

    text = re.sub(r"might've", "might have ", text)

    text = re.sub(r"mightn't", "might not ", text)

    text = re.sub(r"mightn't've", "might not have ", text)

    text = re.sub(r"must've", "must have ", text)

    text = re.sub(r"mustn't", "must not ", text)

    text = re.sub(r"mustn't've", "must not have ", text)

    text = re.sub(r"needn't", "need not ", text)

    text = re.sub(r"needn't've", "need not have ", text)

    text = re.sub(r"o'clock", "of the clock ", text)

    text = re.sub(r"oughtn't", "ought not ", text)

    text = re.sub(r"oughtn't've", "ought not hav ", text)

    text = re.sub(r"shan't", "shall not ", text)

    text = re.sub(r"sha'n't", "shall not ", text)

    text = re.sub(r"shan't've", "shall not have ", text)

    text = re.sub(r"she'd", "she would ", text)

    text = re.sub(r"she'd've", "she would have ", text)

    text = re.sub(r"she'll", "she will ", text)

    text = re.sub(r"she'll've", "she will have ", text)

    text = re.sub(r"she's", "she is ", text)

    text = re.sub(r"should've", "should have ", text)

    text = re.sub(r"shouldn't", "should not ", text)

    text = re.sub(r"shouldn't've", "should not have ", text)

    text = re.sub(r"so've", "so have ", text)

    text = re.sub(r"so have", "so as ", text)

    text = re.sub(r"that'd", "that would ", text)

    text = re.sub(r"that'd've", "that would have ", text)

    text = re.sub(r"that's", "that is ", text)

    text = re.sub(r"there'd", "there would ", text)

    text = re.sub(r"there'd've", "there would have ", text)

    text = re.sub(r"there's", "there is ", text)

    text = re.sub(r"they'd", "they would ", text)

    text = re.sub(r"they'd've", "they would have ", text)

    text = re.sub(r"they'll", "they will ", text)

    text = re.sub(r"they'll've", "they will have ", text)

    text = re.sub(r"they're", "they are ", text)

    text = re.sub(r"they've", "they have ", text)

    text = re.sub(r"to've", "to have ", text)

    text = re.sub(r"wasn't", "was not ", text)

    text = re.sub(r"we'd", "we would ", text)

    text = re.sub(r"we'd've", "we would have ", text)

    text = re.sub(r"we'll", "we will ", text)

    text = re.sub(r"we'll've", "we will have ", text)

    text = re.sub(r"we're", "we are ", text)

    text = re.sub(r"we've", "we have ", text)

    text = re.sub(r"weren't", "were not ", text)

    text = re.sub(r"what'll", "what will ", text)

    text = re.sub(r"what'll've", "what will have ", text)

    text = re.sub(r"what're", "what are ", text)

    text = re.sub(r"what's", "what is ", text)

    text = re.sub(r"when's", "when is ", text)

    text = re.sub(r"when've", "what have ", text)

    text = re.sub(r"where'd", "where did ", text)

    text = re.sub(r"where's", "where is ", text)

    text = re.sub(r"where've", "where have ", text)

    text = re.sub(r"who'll", "who will ", text)

    text = re.sub(r"who'll've", "who will have ", text)

    text = re.sub(r"who's", "who is ", text)

    text = re.sub(r"who've", "who have ", text)

    text = re.sub(r"why's", "why is ", text)

    text = re.sub(r"why've", "why have ", text)

    text = re.sub(r"will've", "will have ", text)

    text = re.sub(r"won't", "will not ", text)

    text = re.sub(r"won't've", "will not have ", text)

    text = re.sub(r"would've", "would have ", text)

    text = re.sub(r"wouldn't", "would not ", text)

    text = re.sub(r"wouldn't've", "would not have ", text)

    text = re.sub(r"y'all", "you all ", text)

    text = re.sub(r"y'all'd", "you all would ", text)

    text = re.sub(r"y'all'd've", "you all would have ", text)

    text = re.sub(r"y'all're", "you all are ", text)

    text = re.sub(r"y'all've", "you all have ", text)

    text = re.sub(r"you'd", "you would ", text)

    text = re.sub(r"you'd've", "you would have ", text)

    text = re.sub(r"you'll", "you will ", text)

    text = re.sub(r"you'll've", "you will have ", text)

    text = re.sub(r"you're", "you are ", text)

    text = re.sub(r"you've", "you have ", text)

    text = re.sub(r"\'s", " ", text)

    text = re.sub(r"\'scuse", " excuse ", text)

    text = re.sub(r"\'\n", " ", text)

    text = re.sub(r"\'\xa0", " ", text)

    text = re.sub('\s+', ' ', text)

    text = text.strip(' ')

    return text
punct = '!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~'
def strip_list_noempty(mylist):

    newlist = (item.strip() if hasattr(item, 'strip') else item for item in mylist)

    return [item for item in newlist if item != '']
def clean_punct(text): 

    words=tokenizer.tokenize(text)

    punctuation_filtered = []

    regex = re.compile('[%s]' % re.escape(punct))

    remove_punctuation = str.maketrans(' ', ' ', punct)

    for w in words:

        if w in tags_features:

            punctuation_filtered.append(w)

        else:

            punctuation_filtered.append(regex.sub('', w))

  

    filtered_list = strip_list_noempty(punctuation_filtered)

        

    return ' '.join(map(str, filtered_list))


def remove_stopwords(text):

    stop_words = set(stopword_list)

    words = tokenizer.tokenize(text)

    filtered = [w for w in words if not w in stop_words]

    return ' '.join(map(str, filtered))
def normalize_corpus(corpus):



    normalized_corpus = []

    

    # normalize each document in the corpus

    for doc in corpus:

        # strip HTML

        doc = strip_html_tags(doc)

        

        # remove accented characters

        doc = remove_accented_chars(doc)

        

        # lowercase the text

        doc = doc.lower()

            

        # expand contraction

        doc = expand_contractions(doc)

        

        #clean punctuations

        doc = clean_punct(doc)

    

        # remove stopwords

        doc = remove_stopwords(doc)



        # lemmatize text

        doc = lemmatize_text(doc)

        

        # remove special characters 

            

        # insert spaces between special characters to isolate them

        special_char_pattern = re.compile(r'([{.(-)!}])')

        doc = special_char_pattern.sub(" \\1 ", doc)

        doc = remove_special_characters(doc)

            

        # remove extra whitespace

        doc = re.sub(' +', ' ', doc)

        

        normalized_corpus.append(doc)



    return normalized_corpus
merged_ques['norm_title'] = normalize_corpus(merged_ques['Title'])
merged_ques['norm_body'] = normalize_corpus(merged_ques['Body'])
no_topics = 20
text = merged_ques['norm_body']
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer_train = TfidfVectorizer(analyzer = 'word',

                                       min_df=0.0,

                                       max_df = 1.0,

                                       strip_accents = None,

                                       encoding = 'utf-8', 

                                       preprocessor=None,

                                       token_pattern=r"(?u)\S\S+", # Need to repeat token pattern

                                       max_features=1000)
TF_IDF_matrix = vectorizer_train.fit_transform(text)
from sklearn.decomposition import LatentDirichletAllocation

lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_method='online', learning_offset=50,random_state=11).fit(TF_IDF_matrix)
def display_topics(model, feature_names, no_top_words):

    for topic_idx, topic in enumerate(model.components_):

        print("--------------------------------------------")

        print("Topic %d:" % (topic_idx))

        print(" ".join([feature_names[i]

                        for i in topic.argsort()[:-no_top_words - 1:-1]]))

        print("--------------------------------------------")

        



no_top_words = 10

display_topics(lda, vectorizer_train.get_feature_names(), no_top_words)
#text[0]
X1 = merged_ques['norm_body']

X2 = merged_ques['norm_title']

y = merged_ques['Tags']


from sklearn.preprocessing import MultiLabelBinarizer

multilabel_binarizer = MultiLabelBinarizer()

y_bin = multilabel_binarizer.fit_transform(y)
vectorizer_X1 = TfidfVectorizer(analyzer = 'word',

                                       min_df=0.0,

                                       max_df = 1.0,

                                       strip_accents = None,

                                       encoding = 'utf-8', 

                                       preprocessor=None,

                                       token_pattern=r"(?u)\S\S+",

                                       max_features=1000)



vectorizer_X2 = TfidfVectorizer(analyzer = 'word',

                                       min_df=0.0,

                                       max_df = 1.0,

                                       strip_accents = None,

                                       encoding = 'utf-8', 

                                       preprocessor=None,

                                       token_pattern=r"(?u)\S\S+",

                                       max_features=1000)
X1.isnull().sum()


X1_tfidf = vectorizer_X1.fit_transform(X1)



X2_tfidf = vectorizer_X2.fit_transform(X2)
from scipy.sparse import hstack

X_tfidf = hstack([X1_tfidf,X2_tfidf])
print('X1', X1.shape)

print('X2', X2.shape)

print('y_bin', y_bin.shape)

print('X_tfidf', X_tfidf.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_bin, test_size = 0.2, random_state = 0)
from sklearn.multiclass import OneVsRestClassifier
def avg_jacard(y_true,y_pred):

    '''

    see https://en.wikipedia.org/wiki/Multi-label_classification#Statistics_and_evaluation_metrics

    '''

    jacard = np.minimum(y_true,y_pred).sum(axis=1) / np.maximum(y_true,y_pred).sum(axis=1)

    

    return jacard.mean()*100



def print_score(y_pred, clf):

    print("Clf: ", clf.__class__.__name__)

    print("Jacard score: {}".format(avg_jacard(y_test, y_pred)))

    print("Hamming loss: {}".format(hamming_loss(y_pred, y_test)*100))

    print("---")    
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import LatentDirichletAllocation

from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.model_selection import train_test_split

from sklearn.model_selection import learning_curve

from sklearn.model_selection import ShuffleSplit

from sklearn.dummy import DummyClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import SGDClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.multiclass import OneVsRestClassifier

from sklearn.svm import LinearSVC

from sklearn.linear_model import Perceptron

from sklearn.linear_model import PassiveAggressiveClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn import model_selection

from sklearn.metrics import make_scorer

from sklearn.metrics import confusion_matrix

from sklearn.metrics import hamming_loss

from sklearn.cluster import KMeans
dummy = DummyClassifier()

sgd = SGDClassifier()

lr = LogisticRegression()

mn = MultinomialNB()

svc = LinearSVC()

perceptron = Perceptron()

pac = PassiveAggressiveClassifier()



for classifier in [dummy, sgd, lr, mn, svc, perceptron, pac]:

    clf = OneVsRestClassifier(classifier)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print_score(y_pred, classifier)
mlpc = MLPClassifier()

mlpc.fit(X_train, y_train)



y_pred = mlpc.predict(X_test)



print_score(y_pred, mlpc)
rfc = RandomForestClassifier()

rfc.fit(X_train, y_train)



y_pred = rfc.predict(X_test)



print_score(y_pred, rfc)
param_grid = {'estimator__C':[1,10,100,1000]

              }
svc = OneVsRestClassifier(LinearSVC())

CV_svc = model_selection.GridSearchCV(estimator=svc, param_grid=param_grid, cv= 5, verbose=10, scoring=make_scorer(avg_jacard,greater_is_better=True))

CV_svc.fit(X_train, y_train)
CV_svc.best_params_
best_model = CV_svc.best_estimator_

best_model
y_pred = best_model.predict(X_test)

print_score(y_pred, best_model)
for i in range(y_train.shape[1]):

    print(multilabel_binarizer.classes_[i])

    print(confusion_matrix(y_test[:,i], y_pred[:,i]))

    print("")
def print_top10(feature_names, clf, class_labels):

    """Prints features with the highest coefficient values, per class"""

    for i, class_label in enumerate(class_labels):

        top10 = np.argsort(clf.coef_[i])[-10:]

        print("--------------------------------------------")

        print("%s: %s" % (class_label,

              " ".join(feature_names[j] for j in top10)))

        print("--------------------------------------------")
feature_names = vectorizer_X1.get_feature_names() + vectorizer_X2.get_feature_names()
print_top10(feature_names, best_model, multilabel_binarizer.classes_)
import pickle



vectorX1_pkl = open('vectorX1.pickle', 'ab')

vectorX2_pkl = open('vectorX2.pickle', 'ab')

mlc_pkl = open('mlc.pickle', 'ab')

      

# source, destination 

pickle.dump(vectorizer_X1, vectorX1_pkl)

pickle.dump(vectorizer_X2, vectorX2_pkl)

pickle.dump(best_model, mlc_pkl)



vectorX1_pkl.close()

vectorX2_pkl.close()

mlc_pkl.close()
#x1_ = pd.Series(['android'])

#x2_ = pd.Series(['activity'])



#x1_tfidf = vectorizer_X1.transform(x1_)

#x2_tfidf = vectorizer_X2.transform(x2_)

#x_tfidf = hstack([x1_tfidf,x2_tfidf])



#y__pred = best_model.predict(x_tfidf)

#y__pred = [for i in y__pred[0]]