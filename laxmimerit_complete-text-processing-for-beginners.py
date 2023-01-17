# pip install -U spacy
# pip install -U spacy-lookups-data
# python -m spacy download en_core_web_sm
# python -m spacy download en_core_web_md
# python -m spacy download en_core_web_lg
import pandas as pd
import numpy as np
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

df = pd.read_csv('../input/sentiment140/training.1600000.processed.noemoticon.csv', encoding = 'latin1', header = None)
df.head()
df = df[[5, 0]]
df.columns = ['twitts', 'sentiment']
df.head()
df['sentiment'].value_counts()
sent_map = {0: 'negative', 4: 'positive'}
df['word_counts'] = df['twitts'].apply(lambda x: len(str(x).split()))
df.head()
df['char_counts'] = df['twitts'].apply(lambda x: len(x))
df.head()
def get_avg_word_len(x):
    words = x.split()
    word_len = 0
    for word in words:
        word_len = word_len + len(word)
    return word_len/len(words) # != len(x)/len(words)
df['avg_word_len'] = df['twitts'].apply(lambda x: get_avg_word_len(x))
len('this is nlp lesson')/4
df.head()
115/19
print(STOP_WORDS)
x = 'this is text data'
x.split()
len([t for t in x.split() if t in STOP_WORDS])


df['stop_words_len'] = df['twitts'].apply(lambda x: len([t for t in x.split() if t in STOP_WORDS]))
df.head()
x = 'this #hashtag and this is @mention'
# x = x.split()
# x
[t for t in x.split() if t.startswith('@')]
    
df['hashtags_count'] = df['twitts'].apply(lambda x: len([t for t in x.split() if t.startswith('#')]))
df['mentions_count'] = df['twitts'].apply(lambda x: len([t for t in x.split() if t.startswith('@')]))
df.head()
df['numerics_count'] = df['twitts'].apply(lambda x: len([t for t in x.split() if t.isdigit()]))
df.head()
df['upper_counts'] = df['twitts'].apply(lambda x: len([t for t in x.split() if t.isupper() and len(x)>3]))
df.head()
df.loc[96]['twitts']

df['twitts'] = df['twitts'].apply(lambda x: x.lower())
df.head(2)
x = "i don't know what you want, can't, he'll, i'd"
contractions = { 
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how does",
"i'd": "i would",
"i'd've": "i would have",
"i'll": "i will",
"i'll've": "i will have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so is",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
" u ": " you ",
" ur ": " your ",
" n ": " and "}
def cont_to_exp(x):
    if type(x) is str:
        for key in contractions:
            value = contractions[key]
            x = x.replace(key, value)
        return x
    else:
        return x
x = "hi, i'd be happy"
cont_to_exp(x)
%%time
df['twitts'] = df['twitts'].apply(lambda x: cont_to_exp(x))
df.head()
import re
x = 'hi my email me at email@email.com another@email.com'
re.findall(r'([a-zA-Z0-9+._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)', x)
df['emails'] = df['twitts'].apply(lambda x: re.findall(r'([a-zA-Z0-9+._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)', x))
df['emails_count'] = df['emails'].apply(lambda x: len(x))
df[df['emails_count']>0].head()
re.sub(r'([a-zA-Z0-9+._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)', '', x)
df['twitts'] = df['twitts'].apply(lambda x: re.sub(r'([a-zA-Z0-9+._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)', '', x))
df[df['emails_count']>0].head()
x = 'hi, to watch more visit https://youtube.com/kgptalkie'
re.findall(r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', x)
df['urls_flag'] = df['twitts'].apply(lambda x: len(re.findall(r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', x)))
re.sub(r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '', x)
df['twitts'] = df['twitts'].apply(lambda x: re.sub(r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '', x))
df.head()
df.loc[0]['twitts']
df['twitts'] = df['twitts'].apply(lambda x: re.sub('RT', "", x))

df['twitts'] = df['twitts'].apply(lambda x: re.sub('[^A-Z a-z 0-9-]+', '', x))
df.head()

x = 'thanks    for    watching and    please    like this video'
" ".join(x.split())
df['twitts'] = df['twitts'].apply(lambda x: " ".join(x.split()))
df.head(2)

from bs4 import BeautifulSoup
x = '<html><h2>Thanks for watching</h2></html>'
BeautifulSoup(x, 'lxml').get_text()
%%time
df['twitts'] = df['twitts'].apply(lambda x: BeautifulSoup(x, 'lxml').get_text())

import unicodedata
x = 'Áccěntěd těxt'
def remove_accented_chars(x):
    x = unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return x
remove_accented_chars(x)
import spacy
x = 'this is stop words removal code is a the an how what'
" ".join([t for t in x.split() if t not in STOP_WORDS])
df['twitts'] = df['twitts'].apply(lambda x: " ".join([t for t in x.split() if t not in STOP_WORDS]))
df.head()

nlp = spacy.load('en_core_web_sm')
x = 'kenichan dived times ball managed save 50 rest'
# dive = dived, time = times, manage = managed
# x = 'i you he she they is am are'
def make_to_base(x):
    x_list = []
    doc = nlp(x)
    
    for token in doc:
        lemma = str(token.lemma_)
        if lemma == '-PRON-' or lemma == 'be':
            lemma = token.text
        x_list.append(lemma)
    print(" ".join(x_list))
        
make_to_base(x)

' '.join(df.head()['twitts'])
text = ' '.join(df['twitts'])
text = text.split()
freq_comm = pd.Series(text).value_counts()
f20 = freq_comm[:20]
f20
df['twitts'] = df['twitts'].apply(lambda x: " ".join([t for t in x.split() if t not in f20]))

rare20 = freq_comm[-20:]
rare20
rare = freq_comm[freq_comm.values == 1]
rare
df['twitts'] = df['twitts'].apply(lambda x: ' '.join([t for t in x.split() if t not in rare20]))
df.head()
# !pip install wordcloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt
%matplotlib inline
x = ' '.join(text[:20000])
len(text)

wc = WordCloud(width = 800, height=400).generate(x)
plt.imshow(wc)
plt.axis('off')
plt.show()

# !pip install -U textblob
# !python -m textblob.download_corpora
from textblob import TextBlob
x = 'tanks forr waching this vidio carri'
x = TextBlob(x).correct()

x

x = 'thanks#watching this video. please like it'
TextBlob(x).words
doc = nlp(x)
for token in doc:
    print(token)
x = 'runs run running ran'
from textblob import Word
for token in x.split():
    print(Word(token).lemmatize())
doc = nlp(x)
for token in doc:
    print(token.lemma_)
x = "Breaking News: Donald Trump, the president of the USA is looking to sign a deal to mine the moon"
doc = nlp(x)
for ent in doc.ents:
    print(ent.text + ' - ' + ent.label_ + ' - ' + str(spacy.explain(ent.label_)))
from spacy import displacy
displacy.render(doc, style = 'ent')
x
for noun in doc.noun_chunks:
    print(noun)
x
tb = TextBlob(x)
tb.detect_language()
tb.translate(to='bn')

from textblob.sentiments import NaiveBayesAnalyzer
x = 'we all stands together to fight with corona virus. we will win together'
tb = TextBlob(x, analyzer=NaiveBayesAnalyzer())
tb.sentiment


x = 'we all are sufering from corona'
tb = TextBlob(x, analyzer=NaiveBayesAnalyzer())
tb.sentiment

x = 'thanks for watching'
tb = TextBlob(x)
tb.ngrams(3)

x = ['this is first sentence this is', 'this is second', 'this is last']
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(ngram_range=(1,1))
text_counts = cv.fit_transform(x)
text_counts.toarray()
cv.get_feature_names()
bow = pd.DataFrame(text_counts.toarray(), columns = cv.get_feature_names())
bow
x

x
bow
bow.shape
tf = bow.copy()
for index, row in enumerate(tf.iterrows()):
    for col in row[1].index:
        tf.loc[index, col] = tf.loc[index, col]/sum(row[1].values)
tf

import numpy as np
x_df = pd.DataFrame(x, columns=['words'])
x_df
bow
N = bow.shape[0]
N
bb = bow.astype('bool')
bb
bb['is'].sum()
cols = bb.columns
cols
nz = []
for col in cols:
    nz.append(bb[col].sum())
nz
idf = []
for index, col in enumerate(cols):
    idf.append(np.log((N + 1)/(nz[index] + 1)) + 1)
idf
bow

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
x_tfidf = tfidf.fit_transform(x_df['words'])
x_tfidf.toarray()
tfidf.idf_
idf

# !python -m spacy download en_core_web_lg
nlp = spacy.load('en_core_web_lg')
doc = nlp('thank you! dog cat lion dfasaa')
for token in doc:
    print(token.text, token.has_vector)
token.vector.shape
nlp('cat').vector.shape
for token1 in doc:
    for token2 in doc:
        print(token1.text, token2.text, token1.similarity(token2))
    print()


df.shape
df0 = df[df['sentiment']==0].sample(2000)
df4 = df[df['sentiment']==4].sample(2000)
dfr = df0.append(df4)
dfr.shape

dfr_feat = dfr.drop(labels=['twitts','sentiment','emails'], axis = 1).reset_index(drop=True)
dfr_feat
y = dfr['sentiment']

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
text_counts = cv.fit_transform(dfr['twitts'])
text_counts.toarray().shape
dfr_bow = pd.DataFrame(text_counts.toarray(), columns=cv.get_feature_names())
dfr_bow.head(2)

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
lgrcv = LogisticRegressionCV(cv = 2, random_state=42, max_iter=1000)
svm = LinearSVC(random_state=42, max_iter=200)
rfc = RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators=200)
clf = {'SGD': sgd, 'LGR': lgr, 'LGR-CV': lgrcv, 'SVM': svm, 'RFC': rfc}
clf.keys()

def classify(X, y):
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)
    
    for key in clf.keys():
        clf[key].fit(X_train, y_train)
        y_pred = clf[key].predict(X_test)
        ac = accuracy_score(y_test, y_pred)
        print(key, " ---> ", ac)

%%time
classify(dfr_bow, y)

dfr_feat.head(2)
%%time
classify(dfr_feat, y)

X = dfr_feat.join(dfr_bow)
%%time
classify(X, y)

from sklearn.feature_extraction.text import TfidfVectorizer
dfr.shape
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(dfr['twitts'])
%%time
classify(pd.DataFrame(X.toarray()), y)

def get_vec(x):
    doc = nlp(x)
    return doc.vector.reshape(1, -1)
%%time
dfr['vec'] = dfr['twitts'].apply(lambda x: get_vec(x))
X = np.concatenate(dfr['vec'].to_numpy(), axis = 0)
X.shape
classify(pd.DataFrame(X), y)

def predict_w2v(x):
    for key in clf.keys():
        y_pred = clf[key].predict(get_vec(x))
        print(key, "-->", y_pred)
predict_w2v('hi, thanks for watching this video. please like and subscribe')
predict_w2v('please let me know if you want more video')
predict_w2v('congratulation looking good congrats')





