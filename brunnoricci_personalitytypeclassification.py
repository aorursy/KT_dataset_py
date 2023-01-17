# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import seaborn as sns
import matplotlib.pyplot as plt
PW = 8
PH = 6
plt.rcParams['figure.figsize'] = (PW, PH) 
plt.rcParams['image.cmap'] = 'gray'

import re
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.options.display.max_rows = 250
pd.options.display.max_columns = 500
pd.options.display.max_colwidth = 500

from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
import string 
import warnings

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/mbti_1.csv", encoding="utf-8")
print("Number of users", len(df))
df.head(3)
#Personality Types
groups = df.groupby("type").count()
groups.sort_values("posts", ascending=False, inplace=True)
print ("Personality types", groups.index.values)

#Priors used below for Random Guessing Estimation
priors = groups["posts"] / groups["posts"].sum()
groups["posts"].plot(kind="bar", title="Number of Users per Personality type");
df["LenPre"] = df["posts"].apply(len)
sns.distplot(df["LenPre"]).set_title("Distribution of Lengths of all 50 Posts");
def preprocess_text(df, remove_special=True):
    #Remove links 
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'https?:\/\/.*?[\s+]', '', x.replace("|"," ") + " "))
    
    #Keep EOS
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'\.', ' EOSTokenDot ', x + " "))
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'\?', ' EOSTokenQuest ', x + " "))
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'!', ' EOSTokenExs ', x + " "))
    
    #Strip Punctation
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'[^\w\s]','',x))

    #Remove Non-words
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'[^a-zA-Z\s]','',x))

    #To lower
    df["posts"] = df["posts"].apply(lambda x: x.lower())

    #Remove multiple letter repating words
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'([a-z])\1{2,}[\s|\w]*','',x)) 

    #Remove short/long words
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'(\b\w{0,3})?\b','',x)) 
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'(\b\w{30,1000})?\b','',x))

    #Remove Personality Types Words
    #This is crutial in order to get valid model accuracy estimation for unseen data. 
    if remove_special:
        pers_types = ['INFP' ,'INFJ', 'INTP', 'INTJ', 'ENTP', 'ENFP', 'ISTP' ,'ISFP' ,'ENTJ', 'ISTJ','ENFJ', 'ISFJ' ,'ESTP', 'ESFP' ,'ESFJ' ,'ESTJ']
        pers_types = [p.lower() for p in pers_types]
        p = re.compile("(" + "|".join(pers_types) + ")")

    df["posts"] = df["posts"].apply(lambda x: p.sub(' PTypeToken ',x))
    return df

#Used for class balancing. When class balancing is used dataset becomes very small.
def subsample(df):
    groups = df.groupby("type").count()
    groups.sort_values("posts", ascending=False, inplace=True)
    
    min_num = groups["posts"][-1]
    min_ind = groups.index[-1]
    ndf = df[df["type"] == min_ind]

    for pt in groups.index[:-1]:
        print(min_num,pt)
        tdf = df[df["type"] == pt].sample(min_num)
        ndf = pd.concat([ndf, tdf])
    return ndf
#Number of Posts per User
df["NumPosts"] = df["posts"].apply(lambda x: len(x.split("|||")))

sns.distplot(df["NumPosts"], kde=False).set_title("Number of Posts per User");
#Split to posts
def extract(posts, new_posts):
    for post in posts[1].split("|||"):
        new_posts.append((posts[0], post))

posts = []
df.apply(lambda x: extract(x, posts), axis=1)
print("Number of users", len(df))
print("Number of posts", len(posts))

df = pd.DataFrame(posts, columns=["type", "posts"])
df["Len"] = df["posts"].apply(len)
sns.distplot(df["Len"]).set_title("Post lengths");
#Preprocess Text
df = preprocess_text(df) 
df["Len"] = df["posts"].apply(len)
sns.distplot(df["Len"]).set_title("Post lengths");
#Remove posts with less than X words
min_words = 15
print("Number of posts", len(df)) 
df["nw"] = df["posts"].apply(lambda x: len(re.findall(r'\w+', x)))
df = df[df["nw"] >= min_words]
print("Number of posts", len(df)) 
df["Len"] = df["posts"].apply(len)
sns.distplot(df["Len"]).set_title("Post lengths");
#Remove long post
max_length = 350
print("Number of posts", len(df)) 
df = df[df["Len"] < 350]
print("Number of posts", len(df)) 
df["Len"] = df["posts"].apply(len)
sns.distplot(df["Len"]).set_title("Post lengths");
#Drop nw Len
df.drop(["nw", "Len"],axis=1, inplace=True)
#Subsample - Used for class balancing. 
#df = subsample(df)
#Stem
stemmer = SnowballStemmer("english")

df["posts"] = df["posts"].apply(lambda x: " ".join(stemmer.stem(p) for p in x.split(" ")))
df.iloc[np.random.choice(len(df),10),:]
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
#Split train/test
vect = CountVectorizer(stop_words='english') 
X =  vect.fit_transform(df["posts"]) 

le = LabelEncoder()
y = le.fit_transform(df["type"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
wdf = pd.DataFrame( vect.get_feature_names(),columns=["word"])
wdf["len"] = wdf.word.apply(len)
sns.distplot(wdf["len"], kde=False).set_title("Words Lentghs Distribution");
#Evaluating Acccuarcy across four categories indipendently 
def cat_accuracy(yp_test, y_test, le):
    ype = np.array(list("".join(le.inverse_transform(yp_test))))
    ye = np.array(list("".join(le.inverse_transform(y_test))))
    return (ype == ye).mean()

def predict_random_guess(priors, lp):
    return np.random.choice(priors.index, lp, p=priors.values)

num_iter = 100
mc16 = np.zeros(num_iter)
mc4 = np.zeros(num_iter)

warnings.filterwarnings(action='ignore', category=DeprecationWarning)
for i in range(100):
    mc16[i] = np.mean(le.transform(predict_random_guess(priors, len(y_test))) == y_test)
    mc4[i] = cat_accuracy(le.transform(predict_random_guess(priors, len(y_test))), y_test, le)


print ("Random Guessing 16 Types:", mc16.mean(), mc16.std())
print ("Random Guessing 4 Categories:", mc4.mean(), mc4.std())
clf = MultinomialNB()
clf.fit(X_train, y_train)

yp_train = clf.predict(X_train)
print("Train Accuracy:", np.mean(yp_train == y_train))

yp_test = clf.predict(X_test)
print("Test Accuracy:", np.mean(yp_test == y_test))
print("******")
print("Categorical Train Accuracy:", cat_accuracy(yp_train, y_train, le))
print("Categorical Test Accuracy:", cat_accuracy(yp_test, y_test, le))
dft = pd.DataFrame(le.inverse_transform(yp_test),columns=["pred"])
dft["cnt"] =  1
dft["same"] = (yp_test == y_test)
dft["same"] = dft["same"].astype(int)

groupsn = dft.groupby("pred").sum()
groupsn.sort_values("cnt", ascending=False, inplace=True)
f, ax = plt.subplots(1,2,figsize=(2*PW,PH))
groupsn["cnt"].plot(kind="bar", title="Distribution of Predicted User Personality Types", ax=ax[0]);
groupsn["same"].plot(kind="bar", title="Distribution of Correctly Classified User Personality Types", ax=ax[1]);
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense,  Dropout, Flatten
from keras.layers import LSTM, Conv1D, Input, MaxPooling1D
from keras.layers.embeddings import Embedding
from sklearn.preprocessing import OneHotEncoder

from scipy import spatial
from sklearn.utils import class_weight
def cat_accu_seq(X_test, y_test, model):
    yp_test = model.predict(X_test)

    yp_test_d =  np.argmax(yp_test, axis=1)
    y_test_d =  np.argmax(y_test, axis=1)
    
    return  cat_accuracy(yp_test_d, y_test_d, le )

def calc_weights(df, le, ohe):
    groups = df.groupby("type").count()
    groups.sort_values("posts", ascending=False, inplace=True)
    
    p = groups["posts"]#.to_dict()
    ohe.transform([[x] for x in le.transform(p.index.values)])
#Prepare X and y
X =  df.posts

le = LabelEncoder()
y = le.fit_transform(df["type"])

ohe = OneHotEncoder(n_values='auto',  sparse=False)
y = ohe.fit_transform(y.reshape(-1, 1))
#Tokenize words
max_nb_words = 200000

tokenizer = Tokenizer(num_words=max_nb_words)
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

#Retokenize
max_nb_words = len(word_index)
tokenizer = Tokenizer(num_words=max_nb_words)
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)
word_index = tokenizer.word_index
#Constants
ptypes_num = 16
max_post_len = np.max([ len(x) for x in sequences])
#Pad Sequen
sequences = sequence.pad_sequences(sequences, maxlen=max_post_len) 
#Split Train/Test
X_train, X_test, y_train, y_test = train_test_split(sequences, y, test_size=0.1, stratify=y, random_state=42)
#Check Stratification
#temp = [np.argmax(x) for x in y_test]
#pd.DataFrame(temp)[0].value_counts().plot(kind="bar");
#Parameters
batch_size = 512
epochs = 5
embedding_vecor_length = 32
lstm_size = 32
#Model
model = Sequential()
model.add(Embedding(max_nb_words, embedding_vecor_length, input_length=max_post_len))
model.add(Dropout(0.25))

model.add(LSTM(lstm_size))
model.add(Dropout(0.25))

model.add(Dense(ptypes_num, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
#Calculate Class Weights
wy = le.fit_transform(df["type"])
cw = class_weight.compute_class_weight('balanced', np.unique(wy), wy)

#Fit Model
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, class_weight=cw, validation_data=(X_test, y_test));
scores_tr = model.evaluate(X_train, y_train, verbose=0)
scores_ts = model.evaluate(X_test, y_test, verbose=0)

print("Train Accuracy:",scores_tr[1])
print("Test Accuracy:", scores_ts[1])
print("******")
print("Categorical Train accuracy:", cat_accu_seq(X_train, y_train, model))
print("Categorical Test Accuracy:", cat_accu_seq(X_test, y_test, model))
yp_test =  np.argmax(model.predict(X_test), axis=1)

dft = pd.DataFrame(le.inverse_transform(yp_test),columns=["pred"])
dft["cnt"] =  1
dft["same"] = (yp_test == np.argmax(y_test, axis=1))
dft["same"] = dft["same"].astype(int)

groupsn = dft.groupby("pred").sum()
groupsn.sort_values("cnt", ascending=False, inplace=True)
f, ax = plt.subplots(1,2,figsize=(2*PW,PH))
groupsn["cnt"].plot(kind="bar", title="Distribution of Predicted User Personality Types", ax=ax[0]);
groupsn["same"].plot(kind="bar", title="Distribution of Correctly Classified User Personality Types", ax=ax[1]);
exit()