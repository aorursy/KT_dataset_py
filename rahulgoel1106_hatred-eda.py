# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



import pandas as pd

import numpy as np

import os



import matplotlib

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

import seaborn as sns



from nltk.corpus import stopwords

from nltk.util import ngrams



from wordcloud import WordCloud



from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA, TruncatedSVD

from sklearn.metrics import classification_report,confusion_matrix



from collections import defaultdict

from collections import Counter

plt.style.use('ggplot')

#stop=set(stopwords.words('english'))



import re

from nltk.tokenize import word_tokenize

import gensim

import string



from tqdm import tqdm

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Embedding, LSTM,Dense, SpatialDropout1D, Dropout

from keras.initializers import Constant

from keras.optimizers import Adam



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/hatred-on-twitter-during-metoo-movement/MeTooHate.csv')

df.shape
#df.info()

#df.describe()

#sns.pairplot(df, vars=["favorite_count","retweet_count","followers_count","friends_count"])
#reference: https://www.kaggle.com/vbmokin/nlp-eda-bag-of-words-tf-idf-glove-bert

# Target class distribution

hate = df[df['category'] == 1].shape[0]

non_hate = df[df['category'] == 0].shape[0]

# Bar plot for 2 classes

plt.rcParams['figure.figsize'] = (7, 5)

plt.bar(10,hate,3, label="Hate", color='red')

plt.bar(15,non_hate,3, label="Non Hate", color='blue')

plt.legend()

plt.ylabel('Sample size')

plt.show()
#Tweet length for different class

def length(text):    

    '''a function which returns the length of text'''

    return len(text)



df['text']=df['text'].astype(str)



df['length'] = df.text.apply(length)



plt.rcParams['figure.figsize'] = (18.0, 6.0)

bins = 150

plt.hist(df[df['category'] == 0]['length'], alpha = 0.6, bins=bins, label='Non Hate')

plt.hist(df[df['category'] == 1]['length'], alpha = 0.8, bins=bins, label='Hate')

plt.xlabel('length')

plt.ylabel('numbers')

plt.legend(loc='upper right')

plt.xlim(0,150)

plt.grid()

plt.show()
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))

tweet_len=df[df['category']==1]['text'].str.len()

ax1.hist(tweet_len,color='blue')

ax1.set_title('Hate tweets')

tweet_len=df[df['category']==0]['text'].str.len()

ax2.hist(tweet_len,color='red')

ax2.set_title('Non Hate tweets')

fig.suptitle('Characters in tweets')

plt.show()
#Average word length

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))

word=df[df['category']==1]['text'].str.split().apply(lambda x : [len(i) for i in x])

sns.distplot(word.map(lambda x: np.mean(x)),ax=ax1,color='blue')

ax1.set_title('Hate')

word=df[df['category']==0]['text'].str.split().apply(lambda x : [len(i) for i in x])

sns.distplot(word.map(lambda x: np.mean(x)),ax=ax2,color='red')

ax2.set_title('Non Hate')

fig.suptitle('Average word length in each tweet')
#Take sample of the data to save computation time

df=df.sample(n=50000)



# Target class distribution

hate = df[df['category'] == 1].shape[0]

non_hate = df[df['category'] == 0].shape[0]

# Bar plot for 2 classes

plt.rcParams['figure.figsize'] = (7, 5)

plt.bar(10,hate,3, label="Hate", color='red')

plt.bar(15,non_hate,3, label="Non Hate", color='blue')

plt.legend()

plt.ylabel('Sample size')

plt.show()
#reference: https://github.com/rahulgoel1106/TwitterDataCleaning/blob/master/TweetClean.ipynb

#Remove urls

def remove_urls(text):

    url_pattern = re.compile(r'https?://\S+|www\.\S+')

    return url_pattern.sub(r'', text)



df["text"] = df["text"].astype(str)

df["text"] = df["text"].apply(lambda text: remove_urls(text))    



#Lower case

df["text"] = df["text"].str.lower()
EMOTICONS = {

    u":‑\)":"Happy face or smiley",

    u":\)":"Happy face or smiley",

    u":-\]":"Happy face or smiley",

    u":\]":"Happy face or smiley",

    u":-3":"Happy face smiley",

    u":3":"Happy face smiley",

    u":->":"Happy face smiley",

    u":>":"Happy face smiley",

    u"8-\)":"Happy face smiley",

    u":o\)":"Happy face smiley",

    u":-\}":"Happy face smiley",

    u":\}":"Happy face smiley",

    u":-\)":"Happy face smiley",

    u":c\)":"Happy face smiley",

    u":\^\)":"Happy face smiley",

    u"=\]":"Happy face smiley",

    u"=\)":"Happy face smiley",

    u":‑D":"Laughing, big grin or laugh with glasses",

    u":D":"Laughing, big grin or laugh with glasses",

    u"8‑D":"Laughing, big grin or laugh with glasses",

    u"8D":"Laughing, big grin or laugh with glasses",

    u"X‑D":"Laughing, big grin or laugh with glasses",

    u"XD":"Laughing, big grin or laugh with glasses",

    u"=D":"Laughing, big grin or laugh with glasses",

    u"=3":"Laughing, big grin or laugh with glasses",

    u"B\^D":"Laughing, big grin or laugh with glasses",

    u":-\)\)":"Very happy",

    u":‑\(":"Frown, sad, andry or pouting",

    u":-\(":"Frown, sad, andry or pouting",

    u":\(":"Frown, sad, andry or pouting",

    u":‑c":"Frown, sad, andry or pouting",

    u":c":"Frown, sad, andry or pouting",

    u":‑<":"Frown, sad, andry or pouting",

    u":<":"Frown, sad, andry or pouting",

    u":‑\[":"Frown, sad, andry or pouting",

    u":\[":"Frown, sad, andry or pouting",

    u":-\|\|":"Frown, sad, andry or pouting",

    u">:\[":"Frown, sad, andry or pouting",

    u":\{":"Frown, sad, andry or pouting",

    u":@":"Frown, sad, andry or pouting",

    u">:\(":"Frown, sad, andry or pouting",

    u":'‑\(":"Crying",

    u":'\(":"Crying",

    u":'‑\)":"Tears of happiness",

    u":'\)":"Tears of happiness",

    u"D‑':":"Horror",

    u"D:<":"Disgust",

    u"D:":"Sadness",

    u"D8":"Great dismay",

    u"D;":"Great dismay",

    u"D=":"Great dismay",

    u"DX":"Great dismay",

    u":‑O":"Surprise",

    u":O":"Surprise",

    u":‑o":"Surprise",

    u":o":"Surprise",

    u":-0":"Shock",

    u"8‑0":"Yawn",

    u">:O":"Yawn",

    u":-\*":"Kiss",

    u":\*":"Kiss",

    u":X":"Kiss",

    u";‑\)":"Wink or smirk",

    u";\)":"Wink or smirk",

    u"\*-\)":"Wink or smirk",

    u"\*\)":"Wink or smirk",

    u";‑\]":"Wink or smirk",

    u";\]":"Wink or smirk",

    u";\^\)":"Wink or smirk",

    u":‑,":"Wink or smirk",

    u";D":"Wink or smirk",

    u":‑P":"Tongue sticking out, cheeky, playful or blowing a raspberry",

    u":P":"Tongue sticking out, cheeky, playful or blowing a raspberry",

    u"X‑P":"Tongue sticking out, cheeky, playful or blowing a raspberry",

    u"XP":"Tongue sticking out, cheeky, playful or blowing a raspberry",

    u":‑Þ":"Tongue sticking out, cheeky, playful or blowing a raspberry",

    u":Þ":"Tongue sticking out, cheeky, playful or blowing a raspberry",

    u":b":"Tongue sticking out, cheeky, playful or blowing a raspberry",

    u"d:":"Tongue sticking out, cheeky, playful or blowing a raspberry",

    u"=p":"Tongue sticking out, cheeky, playful or blowing a raspberry",

    u">:P":"Tongue sticking out, cheeky, playful or blowing a raspberry",

    u":‑/":"Skeptical, annoyed, undecided, uneasy or hesitant",

    u":/":"Skeptical, annoyed, undecided, uneasy or hesitant",

    u":-[.]":"Skeptical, annoyed, undecided, uneasy or hesitant",

    u">:[(\\\)]":"Skeptical, annoyed, undecided, uneasy or hesitant",

    u">:/":"Skeptical, annoyed, undecided, uneasy or hesitant",

    u":[(\\\)]":"Skeptical, annoyed, undecided, uneasy or hesitant",

    u"=/":"Skeptical, annoyed, undecided, uneasy or hesitant",

    u"=[(\\\)]":"Skeptical, annoyed, undecided, uneasy or hesitant",

    u":L":"Skeptical, annoyed, undecided, uneasy or hesitant",

    u"=L":"Skeptical, annoyed, undecided, uneasy or hesitant",

    u":S":"Skeptical, annoyed, undecided, uneasy or hesitant",

    u":‑\|":"Straight face",

    u":\|":"Straight face",

    u":$":"Embarrassed or blushing",

    u":‑x":"Sealed lips or wearing braces or tongue-tied",

    u":x":"Sealed lips or wearing braces or tongue-tied",

    u":‑#":"Sealed lips or wearing braces or tongue-tied",

    u":#":"Sealed lips or wearing braces or tongue-tied",

    u":‑&":"Sealed lips or wearing braces or tongue-tied",

    u":&":"Sealed lips or wearing braces or tongue-tied",

    u"O:‑\)":"Angel, saint or innocent",

    u"O:\)":"Angel, saint or innocent",

    u"0:‑3":"Angel, saint or innocent",

    u"0:3":"Angel, saint or innocent",

    u"0:‑\)":"Angel, saint or innocent",

    u"0:\)":"Angel, saint or innocent",

    u":‑b":"Tongue sticking out, cheeky, playful or blowing a raspberry",

    u"0;\^\)":"Angel, saint or innocent",

    u">:‑\)":"Evil or devilish",

    u">:\)":"Evil or devilish",

    u"\}:‑\)":"Evil or devilish",

    u"\}:\)":"Evil or devilish",

    u"3:‑\)":"Evil or devilish",

    u"3:\)":"Evil or devilish",

    u">;\)":"Evil or devilish",

    u"\|;‑\)":"Cool",

    u"\|‑O":"Bored",

    u":‑J":"Tongue-in-cheek",

    u"#‑\)":"Party all night",

    u"%‑\)":"Drunk or confused",

    u"%\)":"Drunk or confused",

    u":-###..":"Being sick",

    u":###..":"Being sick",

    u"<:‑\|":"Dump",

    u"\(>_<\)":"Troubled",

    u"\(>_<\)>":"Troubled",

    u"\(';'\)":"Baby",

    u"\(\^\^>``":"Nervous or Embarrassed or Troubled or Shy or Sweat drop",

    u"\(\^_\^;\)":"Nervous or Embarrassed or Troubled or Shy or Sweat drop",

    u"\(-_-;\)":"Nervous or Embarrassed or Troubled or Shy or Sweat drop",

    u"\(~_~;\) \(・\.・;\)":"Nervous or Embarrassed or Troubled or Shy or Sweat drop",

    u"\(-_-\)zzz":"Sleeping",

    u"\(\^_-\)":"Wink",

    u"\(\(\+_\+\)\)":"Confused",

    u"\(\+o\+\)":"Confused",

    u"\(o\|o\)":"Ultraman",

    u"\^_\^":"Joyful",

    u"\(\^_\^\)/":"Joyful",

    u"\(\^O\^\)／":"Joyful",

    u"\(\^o\^\)／":"Joyful",

    u"\(__\)":"Kowtow as a sign of respect, or dogeza for apology",

    u"_\(\._\.\)_":"Kowtow as a sign of respect, or dogeza for apology",

    u"<\(_ _\)>":"Kowtow as a sign of respect, or dogeza for apology",

    u"<m\(__\)m>":"Kowtow as a sign of respect, or dogeza for apology",

    u"m\(__\)m":"Kowtow as a sign of respect, or dogeza for apology",

    u"m\(_ _\)m":"Kowtow as a sign of respect, or dogeza for apology",

    u"\('_'\)":"Sad or Crying",

    u"\(/_;\)":"Sad or Crying",

    u"\(T_T\) \(;_;\)":"Sad or Crying",

    u"\(;_;":"Sad of Crying",

    u"\(;_:\)":"Sad or Crying",

    u"\(;O;\)":"Sad or Crying",

    u"\(:_;\)":"Sad or Crying",

    u"\(ToT\)":"Sad or Crying",

    u";_;":"Sad or Crying",

    u";-;":"Sad or Crying",

    u";n;":"Sad or Crying",

    u";;":"Sad or Crying",

    u"Q\.Q":"Sad or Crying",

    u"T\.T":"Sad or Crying",

    u"QQ":"Sad or Crying",

    u"Q_Q":"Sad or Crying",

    u"\(-\.-\)":"Shame",

    u"\(-_-\)":"Shame",

    u"\(一一\)":"Shame",

    u"\(；一_一\)":"Shame",

    u"\(=_=\)":"Tired",

    u"\(=\^\·\^=\)":"cat",

    u"\(=\^\·\·\^=\)":"cat",

    u"=_\^=	":"cat",

    u"\(\.\.\)":"Looking down",

    u"\(\._\.\)":"Looking down",

    u"\^m\^":"Giggling with hand covering mouth",

    u"\(\・\・?":"Confusion",

    u"\(?_?\)":"Confusion",

    u">\^_\^<":"Normal Laugh",

    u"<\^!\^>":"Normal Laugh",

    u"\^/\^":"Normal Laugh",

    u"\（\*\^_\^\*）" :"Normal Laugh",

    u"\(\^<\^\) \(\^\.\^\)":"Normal Laugh",

    u"\(^\^\)":"Normal Laugh",

    u"\(\^\.\^\)":"Normal Laugh",

    u"\(\^_\^\.\)":"Normal Laugh",

    u"\(\^_\^\)":"Normal Laugh",

    u"\(\^\^\)":"Normal Laugh",

    u"\(\^J\^\)":"Normal Laugh",

    u"\(\*\^\.\^\*\)":"Normal Laugh",

    u"\(\^—\^\）":"Normal Laugh",

    u"\(#\^\.\^#\)":"Normal Laugh",

    u"\（\^—\^\）":"Waving",

    u"\(;_;\)/~~~":"Waving",

    u"\(\^\.\^\)/~~~":"Waving",

    u"\(-_-\)/~~~ \($\·\·\)/~~~":"Waving",

    u"\(T_T\)/~~~":"Waving",

    u"\(ToT\)/~~~":"Waving",

    u"\(\*\^0\^\*\)":"Excited",

    u"\(\*_\*\)":"Amazed",

    u"\(\*_\*;":"Amazed",

    u"\(\+_\+\) \(@_@\)":"Amazed",

    u"\(\*\^\^\)v":"Laughing,Cheerful",

    u"\(\^_\^\)v":"Laughing,Cheerful",

    u"\(\(d[-_-]b\)\)":"Headphones,Listening to music",

    u'\(-"-\)':"Worried",

    u"\(ーー;\)":"Worried",

    u"\(\^0_0\^\)":"Eyeglasses",

    u"\(\＾ｖ\＾\)":"Happy",

    u"\(\＾ｕ\＾\)":"Happy",

    u"\(\^\)o\(\^\)":"Happy",

    u"\(\^O\^\)":"Happy",

    u"\(\^o\^\)":"Happy",

    u"\)\^o\^\(":"Happy",

    u":O o_O":"Surprised",

    u"o_0":"Surprised",

    u"o\.O":"Surpised",

    u"\(o\.o\)":"Surprised",

    u"oO":"Surprised",

    u"\(\*￣m￣\)":"Dissatisfied",

    u"\(‘A`\)":"Snubbed or Deflated"

}
#Remove Emojis

# Reference : https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b

def remove_emoji(string):

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', string)





#Remove Emoticons

def remove_emoticons(text):

    emoticon_pattern = re.compile(u'(' + u'|'.join(k for k in EMOTICONS) + u')')

    return emoticon_pattern.sub(r'', text)









#Remove numbers

df["text"] = df["text"].str.replace(r"#(\w+)", '')

df["text"] = df["text"].str.replace(r"@(\w+)", '')

df["text"] = df["text"].apply(lambda text: remove_emoji(text))

df["text"] = df["text"].apply(lambda text: remove_emoticons(text))
import nltk



PUNCT_TO_REMOVE = string.punctuation

def remove_punctuation(text):

    """custom function to remove the punctuation"""

    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))



df["text"] = df["text"].apply(lambda text: remove_punctuation(text))





#Remove stopwords

#nltk.download('stopwords')

from nltk.corpus import stopwords



STOPWORDS = set(stopwords.words('english'))

def remove_stopwords(text):

    """custom function to remove the stopwords"""

    return " ".join([word for word in str(text).split() if word not in STOPWORDS])



df["text"] = df["text"].apply(lambda text: remove_stopwords(text))



#Remove numbers

df["text"] = df["text"].str.replace('\d+', '')
def cv(data):

    count_vectorizer = CountVectorizer()



    emb = count_vectorizer.fit_transform(data)



    return emb, count_vectorizer



list_corpus = df["text"].tolist()

list_labels = df["category"].tolist()



X_train, X_test, y_train, y_test = train_test_split(list_corpus, list_labels, test_size=0.2,random_state=32)



X_train_counts, count_vectorizer = cv(X_train)

X_test_counts = count_vectorizer.transform(X_test)
def plot_LSA(test_data, test_labels, savepath="PCA_demo.csv", plot=True):

        lsa = TruncatedSVD(n_components=2)

        lsa.fit(test_data)

        lsa_scores = lsa.transform(test_data)

        color_mapper = {label:idx for idx,label in enumerate(set(test_labels))}

        color_column = [color_mapper[label] for label in test_labels]

        colors = ['orange','blue']

        if plot:

            plt.scatter(lsa_scores[:,0], lsa_scores[:,1], s=8, alpha=.8, c=test_labels, cmap=matplotlib.colors.ListedColormap(colors))

            orange_patch = mpatches.Patch(color='orange', label='Non Hate')

            blue_patch = mpatches.Patch(color='blue', label='Hate')

            plt.legend(handles=[orange_patch, blue_patch], prop={'size': 30})



fig = plt.figure(figsize=(8, 8))          

plot_LSA(X_train_counts, y_train)

plt.show()

def tfidf(data):

    tfidf_vectorizer = TfidfVectorizer()



    train = tfidf_vectorizer.fit_transform(data)



    return train, tfidf_vectorizer



X_train_tfidf, tfidf_vectorizer = tfidf(X_train)

X_test_tfidf = tfidf_vectorizer.transform(X_test)
fig = plt.figure(figsize=(8, 8))          

plot_LSA(X_train_tfidf, y_train)

plt.show()
def create_corpus_new(df):

    corpus=[]

    for tweet in tqdm(df['text']):

        words=[word.lower() for word in word_tokenize(tweet)]

        corpus.append(words)

    return corpus  
corpus=create_corpus_new(df)
embedding_dict={}

with open('../input/privatedata/glove.twitter.27B.25d.txt','r') as f:

    for line in f:

        values=line.split()

        word = values[0]

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

embedding_matrix=np.zeros((num_words,25))



for word,i in tqdm(word_index.items()):

    if i < num_words:

        emb_vec=embedding_dict.get(word)

        if emb_vec is not None:

            embedding_matrix[i]=emb_vec 
#Baseline Model with GloVe results¶

model=Sequential()



embedding=Embedding(num_words,25,embeddings_initializer=Constant(embedding_matrix),

                   input_length=MAX_LEN,trainable=False)



model.add(embedding)

model.add(SpatialDropout1D(0.2))

model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(1, activation='sigmoid'))





optimzer=Adam(learning_rate=3e-4)



model.compile(loss='binary_crossentropy',optimizer=optimzer,metrics=['accuracy'])
train=tweet_pad[:df.shape[0]]

test=tweet_pad[df.shape[0]:]
X_train,X_test,y_train,y_test=train_test_split(train,df['category'].values,test_size=0.2)

print('Shape of train',X_train.shape)

print("Shape of Validation ",X_test.shape)
history=model.fit(X_train,y_train,batch_size=1000,epochs=10,validation_data=(X_test,y_test),verbose=2)
#Vectorizing with TF-IDF Vectorizer and creating feature matrix

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5,

                        ngram_range=(1, 3), 

                        stop_words='english')



# We transform each tweet into a vector

features = tfidf.fit_transform(df.text).toarray()



labels = df.category



print("Each of the %d tweet is represented by %d features (TF-IDF score of unigrams, bigrams and trigram)" %(features.shape))
from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import LinearSVC

from sklearn.model_selection import cross_val_score



models = [

    RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0),

    LinearSVC(),

    MultinomialNB()

]



# 5 Cross-validation

CV = 10

cv_df = pd.DataFrame(index=range(CV * len(models)))



entries = []

for model in models:

  model_name = model.__class__.__name__

  accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)

  print(model)

  for fold_idx, accuracy in enumerate(accuracies):

    entries.append((model_name, fold_idx, accuracy))

    

cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
mean_accuracy = cv_df.groupby('model_name').accuracy.mean()

std_accuracy = cv_df.groupby('model_name').accuracy.std()



acc = pd.concat([mean_accuracy, std_accuracy], axis= 1, 

          ignore_index=True)

acc.columns = ['Mean Accuracy', 'Standard deviation']

acc
plt.figure(figsize=(8,5))

sns.boxplot(x='model_name', y='accuracy', 

            data=cv_df, 

            color='lightblue', 

            showmeans=True)

plt.title("MEAN ACCURACY (cv = 10)\n", size=14);