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



df=pd.read_csv('/kaggle/input/sentiment140/training.1600000.processed.noemoticon.csv', encoding = 'latin1', header = None)
df = df[[5, 0]]
df.columns = ['Tweets', 'Sentiments']
set_map = {0: 'Negative', 4: 'Positive'}
df['Word_counts'] = df['Tweets'].apply(lambda x: len(str(x).split()))
df['Character_counts'] = df['Tweets'].apply(lambda x: len(x))
def get_avg_word_len(x):

    words = x.split()

    word_len = 0

    for word in words:

        word_len = word_len + len(word)

    return word_len/len(words)
df['Avg_word_len'] = df['Tweets'].apply(lambda x: get_avg_word_len(x))
import spacy

from spacy.lang.en.stop_words import STOP_WORDS
df['stop_words'] = df['Tweets'].apply(lambda x: len([t for t in x.split() if t in STOP_WORDS]))
df['hashtags_count'] = df['Tweets'].apply(lambda x: len([t for t in x.split() if t.startswith('#')]))

df['mention_count'] = df['Tweets'].apply(lambda x: len([t for t in x.split() if t.startswith('@')]))
df['numerics_count'] = df['Tweets'].apply(lambda x: len([t for t in x.split() if t.isdigit()]))
df['UPPER_CASE_COUNT'] = df['Tweets'].apply(lambda x: len([t for t in  x.split() if t.isupper() and len(x)>3]))
contractions = {

"aight": "alright",

"ain't": "am not",

"amn't": "am not",

"aren't": "are not",

"can't": "can not",

"cause": "because",

"could've": "could have",

"couldn't": "could not",

"couldn't've": "could not have",

"daren't": "dare not",

"daresn't": "dare not",

"dasn't": "dare not",

"didn't": "did not",

"doesn't": "does not",

"don't": "do not",

"d'ye": "do you",

"e'er": "ever",

"everybody's": "everybody is",

"everyone's": "everyone is",

"finna": "fixing to",

"g'day": "good day",

"gimme": "give me",

"giv'n": "given",

"gonna": "going to",

"gon't": "go not",

"gotta": "got to",

"hadn't": "had not",

"had've": "had have",

"hasn't": "has not",

"haven't": "have not",

"he'd": "he had",

"he'dn't've'd": "he would not have had",

"he'll": "he will",

"he's": "he is",

"he've": "he have",

"how'd": "how would",

"howdy": "how do you do",

"how'll": "how will",

"how're": "how are",

"I'll": "I will",

"I'm": "I am",

"I'm'a": "I am about to",

"I'm'o": "I am going to",

"innit": "is it not",

"I've": "I have",

"isn't": "is not",

"it'd": "it would",

"it'll": "it will",

"it's": "it is",

"let's": "let us",

"ma'am": "madam",

"mayn't": "may not",

"may've": "may have",

"methinks": "me thinks",

"mightn't": "might not",

"might've": "might have",

"mustn't": "must not",

"mustn't've": "must not have",

"must've": "must have",

"needn't": "need not",

"ne'er": "never",

"o'clock": "of the clock",

"o'er": "over",

"ol'": "old",

"oughtn't": "ought not",

"'s": "is",

"shalln't": "shall not",

"shan't": "shall not",

"she'd": "she would",

"she'll": "she shall",

"she'll": "she will",

"she's": "she has",

"she's": "she is",

"should've": "should have",

"shouldn't": "should not",

"shouldn't've": "should not have",

"somebody's": "somebody has",

"somebody's": "somebody is",

"someone's": "someone has",

"someone's": "someone is",

"something's": "something has",

"something's": "something is",

"so're": "so are",

"that'll": "that shall",

"that'll": "that will",

"that're": "that are",

"that's": "that has",

"that's": "that is",

"that'd": "that would",

"that'd": "that had",

"there'd": "there had",

"there'd": "there would",

"there'll": "there shall",

"there'll": "there will",

"there're": "there are",

"there's": "there has",

"there's": "there is",

"these're": "these are",

"these've": "these have",

"they'd": "they had",

"they'd": "they would",

"they'll": "they shall",

"they'll": "they will",

"they're": "they are",

"they're": "they were",

"they've": "they have",

"this's": "this has",

"this's": "this is",

"those're": "those are",

"those've": "those have",

"'tis": "it is",

"to've": "to have",

"'twas": "it was",

"wanna": "want to",

"wasn't": "was not",

"we'd": "we had",

"we'd": "we would",

"we'd": "we did",

"we'll": "we shall",

"we'll": "we will",

"we're": "we are",

"we've": "we have",

"weren't": "were not",

"what'd": "what did",

"what'll": "what shall",

"what'll": "what will",

"what're": "what are",

"what're": "what were",

"what's": "what has",

"what's": "what is",

"what's": "what does",

"what've": "what have",

"when's": "when has",

"when's": "when is",

"where'd": "where did",

"where'll": "where shall",

"where'll": "where will",

"where're": "where are",

"where's": "where has",

"where's": "where is",

"where's": "where does",

"where've": "where have",

"which'd": "which had",

"which'd": "which would",

"which'll": "which shall",

"which'll": "which will",

"which're": "which are",

"which's": "which has",

"which's": "which is",

"which've": "which have",

"who'd": "who would",

"who'd": "who had",

"who'd": "who did",

"who'd've": "who would have",

"who'll": "who shall",

"who'll": "who will",

"who're": "who are",

"who's": "who has",

"who's": "who is",

"who's": "who does",

"who've": "who have",

"why'd": "why did",

"why're": "why are",

"why's": "why has",

"why's": "why is",

"why's": "why does",

"won't": "will not",

"would've": "would have",

"wouldn't": "would not",

"wouldn't've": "would not have",

"y'all": "you all",

"y'all'd've": "you all would have",

"y'all'dn't've'd": "you all would not have had",

"y'all're": "you all are",

"you'd": "you had",

"you'd": "you would",

"you'll": "you shall",

"you'll": "you will",

"you're": "you are",

"you're": "you are",

"you've": "you have",

" u ": "you",

" ur ": "your",

" n ": "and"

}
def cont_to_exp(x):

    if type(x) is str:

        for key in contractions:

            value = contractions[key]

            x = x.replace(key,value)

        return x

    else:

        return x
df['Tweets'] = df['Tweets'].apply(lambda x: cont_to_exp(x))
import re
df['Emails'] = df['Tweets'].apply(lambda x: re.findall(r'([a-zA-Z0-9+._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)',x))
df['Emails_count'] = df['Emails'].apply(lambda x: len(x))
df['Tweets'] = df['Tweets'].apply(lambda x: re.sub(r'([a-zA-Z0-9+._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)', '',x))
df['URL_Flags'] = df['Tweets'].apply(lambda x: len(re.findall(r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', x)))
df['Tweets'] = df['Tweets'].apply(lambda x: re.sub(r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '', x))
df['Tweets'] = df['Tweets'].apply(lambda x: re.sub('RT', '', x))
df['Tweets'] = df['Tweets'].apply(lambda x: re.sub('[^a-z A-Z 0-9-]+', '', x))
df['Tweets'] = df['Tweets'].apply(lambda x: ' '.join(x.split()))
from bs4 import BeautifulSoup
df['Tweets'] = df['Tweets'].apply(lambda x: BeautifulSoup(x, 'lxml').get_text())
import spacy
df['Tweets'] = df['Tweets'].apply(lambda x: ' '.join([t for t in x.split() if t not in STOP_WORDS]))
text = ' '.join(df['Tweets'])
text = text.split()
freq_comm = pd.Series(text).value_counts()
f_20 = freq_comm[:20]
df['Tweets'] = df['Tweets'].apply(lambda x: " ".join([t for t in x.split() if t not in f_20]))
rare_20 = freq_comm[-20:]
df['Tweets'] = df['Tweets'].apply(lambda x: ' '.join([t for t in x.split() if t not in rare_20]))
from wordcloud import WordCloud

import matplotlib.pyplot as plt

%matplotlib inline
x = ' '.join(text[:20000])
wc = WordCloud(width = 800, height = 400).generate(x)

plt.imshow(wc)

plt.axis('off')

plt.show()
df_0 = df[df['Sentiments'] == 0].sample(2000)

df_4 = df[df['Sentiments'] == 4].sample(2000)
dfr = df_0.append(df_4)
dfr_feat = dfr.drop(labels = ['Tweets', 'Sentiments', 'Emails'], axis = 1).reset_index(drop = True)
y = dfr['Sentiments']
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

text_counts = cv.fit_transform(dfr['Tweets'])
dfr_bow = pd.DataFrame(text_counts.toarray(), columns = cv.get_feature_names())
from sklearn.linear_model import SGDClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LogisticRegressionCV

from sklearn.svm import LinearSVC

from sklearn.ensemble import RandomForestClassifier



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.preprocessing import MinMaxScaler
sgd = SGDClassifier(n_jobs=-1, random_state=42, max_iter=200)

lgr = LogisticRegression(random_state=42, max_iter=200)

lgr_cv = LogisticRegressionCV(cv=2, random_state=42, max_iter=1000)

svm = LinearSVC(random_state=42, max_iter=200)

rfc = RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators=200)
clf = {'SGD': sgd, 'LGR': lgr, 'LGR_CV': lgr_cv, 'SVM': svm, 'RFC': rfc}
def classify(X,y):

    scaler = MinMaxScaler(feature_range=(0,1))

    X = scaler.fit_transform(X)

    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

    

    for key in clf.keys():

        clf[key].fit(X_train, y_train)

        y_pred = clf[key].predict(X_test)

        ac = accuracy_score(y_test, y_pred)

        print(key, ' ---> ' , ac)
classify(dfr_bow, y)
classify(dfr_feat, y)
X = dfr_feat.join(dfr_bow)
classify(X, y)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()

X = tfidf.fit_transform(dfr['Tweets'])
classify(pd.DataFrame(X.toarray()), y)
nlp = spacy.load('en_core_web_lg')
def get_vec(x):

    doc = nlp(x)

    return doc.vector.reshape(1, -1)
dfr['vec'] = dfr['Tweets'].apply(lambda x: get_vec(x))
X= np.concatenate(dfr['vec'].to_numpy(), axis=0)
classify(pd.DataFrame(X), y)