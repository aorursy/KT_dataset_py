# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# Any results you write to the current directory are saved as output.
# NLTK

import nltk

from collections import Counter

nltk.download('punkt')

nltk.download('averaged_perceptron_tagger')

nltk.download('wordnet')

nltk.download('brown')

nltk.download('names')





# Tokeenization

from nltk import sent_tokenize, word_tokenize

from nltk.corpus import stopwords

from string import punctuation

from nltk.corpus import wordnet

from nltk import pos_tag, pos_tag_sents

from nltk.stem import WordNetLemmatizer



# Vectorizer

from io import StringIO

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer



# Utilities

from collections import OrderedDict

import pickle

from tqdm import tqdm

import operator 



# Plotting

from matplotlib import pyplot as plt



# Classifier

from sklearn.naive_bayes import MultinomialNB



# Word Embedding

from gensim.models.keyedvectors import KeyedVectors



# Spell check



import re

import gc



# Sklearn

from sklearn.model_selection import train_test_split, KFold

from sklearn.metrics import classification_report, accuracy_score, recall_score, f1_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.decomposition import TruncatedSVD

from sklearn.naive_bayes import GaussianNB, MultinomialNB



from gensim.matutils import unitvec



import logging

import normalise



training_dataset = pd.read_csv("../input/train.csv")

testing_dataset = pd.read_csv("../input/test.csv")
stopwords_en = set(stopwords.words('english'))

stopwords_en_withpunct = stopwords_en.union(set(punctuation))

#stopwords_json = {"en":["a","a's","able","about","above","according","accordingly","across","actually","after","afterwards","again","against","ain't","all","allow","allows","almost","alone","along","already","also","although","always","am","among","amongst","an","and","another","any","anybody","anyhow","anyone","anything","anyway","anyways","anywhere","apart","appear","appreciate","appropriate","are","aren't","around","as","aside","ask","asking","associated","at","available","away","awfully","b","be","became","because","become","becomes","becoming","been","before","beforehand","behind","being","believe","below","beside","besides","best","better","between","beyond","both","brief","but","by","c","c'mon","c's","came","can","can't","cannot","cant","cause","causes","certain","certainly","changes","clearly","co","com","come","comes","concerning","consequently","consider","considering","contain","containing","contains","corresponding","could","couldn't","course","currently","d","definitely","described","despite","did","didn't","different","do","does","doesn't","doing","don't","done","down","downwards","during","e","each","edu","eg","eight","either","else","elsewhere","enough","entirely","especially","et","etc","even","ever","every","everybody","everyone","everything","everywhere","ex","exactly","example","except","f","far","few","fifth","first","five","followed","following","follows","for","former","formerly","forth","four","from","further","furthermore","g","get","gets","getting","given","gives","go","goes","going","gone","got","gotten","greetings","h","had","hadn't","happens","hardly","has","hasn't","have","haven't","having","he","he's","hello","help","hence","her","here","here's","hereafter","hereby","herein","hereupon","hers","herself","hi","him","himself","his","hither","hopefully","how","howbeit","however","i","i'd","i'll","i'm","i've","ie","if","ignored","immediate","in","inasmuch","inc","indeed","indicate","indicated","indicates","inner","insofar","instead","into","inward","is","isn't","it","it'd","it'll","it's","its","itself","j","just","k","keep","keeps","kept","know","known","knows","l","last","lately","later","latter","latterly","least","less","lest","let","let's","like","liked","likely","little","look","looking","looks","ltd","m","mainly","many","may","maybe","me","mean","meanwhile","merely","might","more","moreover","most","mostly","much","must","my","myself","n","name","namely","nd","near","nearly","necessary","need","needs","neither","never","nevertheless","new","next","nine","no","nobody","non","none","noone","nor","normally","not","nothing","novel","now","nowhere","o","obviously","of","off","often","oh","ok","okay","old","on","once","one","ones","only","onto","or","other","others","otherwise","ought","our","ours","ourselves","out","outside","over","overall","own","p","particular","particularly","per","perhaps","placed","please","plus","possible","presumably","probably","provides","q","que","quite","qv","r","rather","rd","re","really","reasonably","regarding","regardless","regards","relatively","respectively","right","s","said","same","saw","say","saying","says","second","secondly","see","seeing","seem","seemed","seeming","seems","seen","self","selves","sensible","sent","serious","seriously","seven","several","shall","she","should","shouldn't","since","six","so","some","somebody","somehow","someone","something","sometime","sometimes","somewhat","somewhere","soon","sorry","specified","specify","specifying","still","sub","such","sup","sure","t","t's","take","taken","tell","tends","th","than","thank","thanks","thanx","that","that's","thats","the","their","theirs","them","themselves","then","thence","there","there's","thereafter","thereby","therefore","therein","theres","thereupon","these","they","they'd","they'll","they're","they've","think","third","this","thorough","thoroughly","those","though","three","through","throughout","thru","thus","to","together","too","took","toward","towards","tried","tries","truly","try","trying","twice","two","u","un","under","unfortunately","unless","unlikely","until","unto","up","upon","us","use","used","useful","uses","using","usually","uucp","v","value","various","very","via","viz","vs","w","want","wants","was","wasn't","way","we","we'd","we'll","we're","we've","welcome","well","went","were","weren't","what","what's","whatever","when","whence","whenever","where","where's","whereafter","whereas","whereby","wherein","whereupon","wherever","whether","which","while","whither","who","who's","whoever","whole","whom","whose","why","will","willing","wish","with","within","without","won't","wonder","would","wouldn't","x","y","yes","yet","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves","z","zero"]}

#stopwords_combined = set.union(set(stopwords_json['en']), stopwords_en_withpunct)

stopwords = stopwords_en_withpunct

stopwords = stopwords.union(set(['']))
wnl = WordNetLemmatizer()



def penn2morphy(penntag):

    """ Converts Penn Treebank tags to WordNet. """

    morphy_tag = {'NN':'n', 'JJ':'a',

                  'VB':'v', 'RB':'r'}

    try:

        return morphy_tag[penntag[:2]]

    except:

        return 'n' 

    

def lemmatize_sent(text): 

    # Text input is string, returns lowercased strings.

    return [wnl.lemmatize(word.lower(), pos=penn2morphy(tag)) 

            for word, tag in pos_tag(word_tokenize(text))]


def preprocess_text(text):

    # Input: str, i.e. document/sentence

    # Output: list(str) , i.e. list of lemmas

    

    tokens =  [word for word in lemmatize_sent(text) 

            if word not in stopwords

            and not word.isdigit()]

    try: 

        tokens = normalise.normalise(text=tokens, user_abbrevs=custom_dictionary, verbose=False)

    except:

        result = []

        for text in tokens:

            try:

                result.append(normalise.normalise(texts, verbose=False))

            except:

                result.append(text)

        tokens = result

    tokens = [word for word in tokens

          if word not in stopwords]

    return tokens;
class CustomVectorizer(TfidfVectorizer):

    def build_analyzer(self):

        stop_words = self.get_stop_words()

    

        def analyser(doc):

            if (self.lowercase == True):

                doc = doc.lower()

            tokens = preprocess_text(doc)

            

            return(self._word_ngrams(tokens, stop_words))

        return (analyser)
max_features_TFIDF = 5000

ngram_range_TFIDF = (1, 1)
tfidf_vectorizer_preprocessing = CustomVectorizer(stop_words=stopwords,

                                    ngram_range=ngram_range_TFIDF,

                                    max_features=max_features_TFIDF,

                                    encoding='utf-8',

                                    decode_error='strict',

                                    strip_accents = None,

                                    lowercase=True)
train_vectors = tfidf_vectorizer_preprocessing.fit_transform(training_dataset['question_text'])

test_vectors = tfidf_vectorizer_preprocessing.transform(testing_dataset['question_text'])
randomForestClassifier = RandomForestClassifier(n_estimators=30, 

                       max_depth=120,

                       min_samples_leaf=2,

                       max_features="auto",

                       n_jobs=-1)

X_training = train_vectors

Y_training = training_dataset['target']

randomForestClassifier.fit(X_training, Y_training)

X_test = test_vectors

Y_predict = randomForestClassifier.predict(X_test)

data_submission = pd.DataFrame({'qid': testing_dataset.qid, 'prediction': Y_predict})

data_submission.to_csv("submission.csv", index=False)