# Importing libraries and 

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# importing data

df = pd.read_csv("../input/scripts.csv")

del df["Unnamed: 0"]



df.head()
dial_df = df.drop(["EpisodeNo","SEID","Season"],axis=1)

dial_df.head()
dial_df["Character"].value_counts().head(12).plot(kind="bar")
def corpus_creator(name):

    st = "" 

    for i in dial_df["Dialogue"][dial_df["Character"]==name]:

        st = st + i

    return st



corpus_df = pd.DataFrame()

corpus_df["Character"] = list(dial_df["Character"].value_counts().head(12).index)



li = []

for i in corpus_df["Character"]:

    li.append(corpus_creator(i))



corpus_df["Dialogues"] = li



corpus_df
from sklearn.feature_extraction import text

punc = ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}',"%"]

stop_words = text.ENGLISH_STOP_WORDS.union(punc)
from nltk.tokenize import word_tokenize

def text_processor(dialogue):

    dialogue = word_tokenize(dialogue)

    nopunc=[word.lower() for word in dialogue if word not in stop_words]

    nopunc=' '.join(nopunc)

    return [word for word in nopunc.split()]
corpus_df["Dialogues"] = corpus_df["Dialogues"].apply(lambda x: text_processor(x))

corpus_df
corpus_df["Length"] = corpus_df["Dialogues"].apply(lambda x: len(x))

corpus_df
fig, ax = plt.subplots(figsize=(10,10))

sns.barplot(ax=ax,y="Length",x="Character",data=corpus_df)
import gensim

# Creating a dictionary for mapping every word to a number

dictionary = gensim.corpora.Dictionary(corpus_df["Dialogues"])

print(dictionary[567])

print(dictionary.token2id['cereal'])

print("Number of words in dictionary: ",len(dictionary))



# Now, we create a corpus which is a list of bags of words. A bag-of-words representation for a document just lists the number of times each word occurs in the document.

corpus = [dictionary.doc2bow(bw) for bw in corpus_df["Dialogues"]]



# Now, we use tf-idf model on our corpus

tf_idf = gensim.models.TfidfModel(corpus)



# Creating a Similarity objectr

sims = gensim.similarities.Similarity('',tf_idf[corpus],num_features=len(dictionary))



# Creating a dataframe out of similarities

sim_list = []

for i in range(12):

    query = dictionary.doc2bow(corpus_df["Dialogues"][i])

    query_tf_idf = tf_idf[query]

    sim_list.append(sims[query_tf_idf])

    

corr_df = pd.DataFrame()

j=0

for i in corpus_df["Character"]:

    corr_df[i] = sim_list[j]

    j = j + 1   
fig, ax = plt.subplots(figsize=(12,12))

sns.heatmap(corr_df,ax=ax,annot=True)

ax.set_yticklabels(corpus_df.Character)

plt.savefig('similarity.png')

plt.show()
dial_df = dial_df[(dial_df["Character"]=="ELAINE") | (dial_df["Character"]=="GEORGE") | (dial_df["Character"]=="KRAMER")]

dial_df.head(8)
def text_process(dialogue):

    nopunc=[word.lower() for word in dialogue if word not in stop_words]

    nopunc=''.join(nopunc)

    return [word for word in nopunc.split()]
X = dial_df["Dialogue"]

y = dial_df["Character"]
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(analyzer=text_process).fit(X)
print(len(vectorizer.vocabulary_))

X = vectorizer.transform(X)

# Splitting the data into train and test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)
from sklearn.naive_bayes import MultinomialNB as MNB

from sklearn.linear_model import LogisticRegression as LR

from sklearn.ensemble import RandomForestClassifier as RFC

from sklearn.ensemble import VotingClassifier as VC

mnb = MNB(alpha=10)

lr = LR(random_state=101)

rfc = RFC(n_estimators=80, criterion="entropy", random_state=42, n_jobs=-1)

clf = VC(estimators=[('mnb', mnb), ('lr', lr), ('rfc', rfc)], voting='hard')
# Fitting and predicting

clf.fit(X_train,y_train)



predict = clf.predict(X_test)
# Classification report

from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test, predict))

print('\n')

print(classification_report(y_test, predict))
def predictor(s):

    s = vectorizer.transform(s)

    pre = clf.predict(s)

    print(pre)
# Answer should be Kramer

predictor(['I\'m on the Mexican, whoa oh oh, radio.'])
# Answer should be Elaine

predictor(['Do you have any idea how much time I waste in this apartment?'])
# Answer should be George 

predictor(['Yeah. I figured since I was lying about my income for a couple of years, I could afford a fake house in the Hamptons.'])
# Now, a random sentence

predictor(['Jerry, I\'m gonna go join the circus.'])
# A random sentence

predictor(['I wish we can find some way to move past this.'])
# Answer should be Kramer

predictor(['You’re becoming one of the glitterati.'])
# Answer should be Elaine

predictor(['Jerry, we have to have sex to save the friendship.'])
import keras

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout

from keras.preprocessing.text import Tokenizer
# Encoding categorical data using label encoding and one-hot encoding 

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X_1 = LabelEncoder()

y = labelencoder_X_1.fit_transform(y)

y = y.reshape(-1,1)

onehotencoder = OneHotEncoder(categorical_features = [0])

y = onehotencoder.fit_transform(y).toarray()

# This would transform the dependent variable into a 3-column matrix, first for Elaine, second for George and third for Kramer 
y = np.delete(y,2,1).astype(int)
# Splitting the data again into train and test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)
max_words = 14178

num_classes = 2



model = Sequential()

model.add(Dense(512, input_shape=(max_words,)))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(256, input_shape=(max_words,)))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(128, input_shape=(max_words,)))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes))

model.add(Activation('softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history=model.fit(X_train, y_train, epochs=60, batch_size=256)
acc = history.history['acc']

loss = history.history['loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')

plt.title('Training accuracy')

plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')

plt.title('Training loss')

plt.legend()

plt.show()
evaluation=model.evaluate(X_test,y_test, batch_size=256)

print('loss =', evaluation[0])

print('accuracy =', evaluation[1])