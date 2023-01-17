import matplotlib.pyplot as plt

import seaborn as sns

import random as rnd

import re

import string

import operator

import numpy as np

import pandas as pd



from nltk.corpus import stopwords

from nltk.tokenize import TweetTokenizer, word_tokenize





from keras.models import Sequential

from keras.initializers import Constant

from keras.layers import (LSTM, 

                          Embedding, 

                          BatchNormalization,

                          SpatialDropout1D,

                          Dense, 

                          Dropout)



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from keras.optimizers import Adam



from sklearn.model_selection import train_test_split

from sklearn.metrics import (

    precision_score, 

    recall_score, 

    f1_score, 

    classification_report,

    accuracy_score)



import tensorflow as tf

from clr_callback import *
rnd.seed(42) 

np.random.seed(42)

tf.random.set_seed(42)
df = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

df_test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')



df.shape, df_test.shape
df.loc[df['text'] == 'like for the music video I want some real action shit like burning buildings and police chases not some weak ben winston shit', 'target'] = 0

df.loc[df['text'] == 'Hellfire is surrounded by desires so be careful and donÛªt let your desires control you! #Afterlife', 'target'] = 0

df.loc[df['text'] == 'To fight bioterrorism sir.', 'target'] = 0

df.loc[df['text'] == '.POTUS #StrategicPatience is a strategy for #Genocide; refugees; IDP Internally displaced people; horror; etc. https://t.co/rqWuoy1fm4', 'target'] = 1

df.loc[df['text'] == 'CLEARED:incident with injury:I-495  inner loop Exit 31 - MD 97/Georgia Ave Silver Spring', 'target'] = 1

df.loc[df['text'] == '#foodscare #offers2go #NestleIndia slips into loss after #Magginoodle #ban unsafe and hazardous for #humanconsumption', 'target'] = 0

df.loc[df['text'] == 'In #islam saving a person is equal in reward to saving all humans! Islam is the opposite of terrorism!', 'target'] = 0

df.loc[df['text'] == 'Who is bringing the tornadoes and floods. Who is bringing the climate change. God is after America He is plaguing her\n \n#FARRAKHAN #QUOTE', 'target'] = 1

df.loc[df['text'] == 'RT NotExplained: The only known image of infamous hijacker D.B. Cooper. http://t.co/JlzK2HdeTG', 'target'] = 1

df.loc[df['text'] == "Mmmmmm I'm burning.... I'm burning buildings I'm building.... Oooooohhhh oooh ooh...", 'target'] = 0

df.loc[df['text'] == "wowo--=== 12000 Nigerian refugees repatriated from Cameroon", 'target'] = 0

df.loc[df['text'] == "He came to a land which was engulfed in tribal war and turned it into a land of peace i.e. Madinah. #ProphetMuhammad #islam", 'target'] = 0

df.loc[df['text'] == "Hellfire! We donÛªt even want to think about it or mention it so letÛªs not do anything that leads to it #islam!", 'target'] = 0

df.loc[df['text'] == "The Prophet (peace be upon him) said 'Save yourself from Hellfire even if it is by giving half a date in charity.'", 'target'] = 0

df.loc[df['text'] == "Caution: breathing may be hazardous to your health.", 'target'] = 1

df.loc[df['text'] == "I Pledge Allegiance To The P.O.P.E. And The Burning Buildings of Epic City. ??????", 'target'] = 0

df.loc[df['text'] == "#Allah describes piling up #wealth thinking it would last #forever as the description of the people of #Hellfire in Surah Humaza. #Reflect", 'target'] = 0

df.loc[df['text'] == "that horrible sinking feeling when youÛªve been at home on your phone for a while and you realise its been on 3G this whole time", 'target'] = 0
def clean(tweet):             

    # Special characters

    tweet = re.sub(r"\x89Û_", "", tweet)

    tweet = re.sub(r"\x89ÛÒ", "", tweet)

    tweet = re.sub(r"\x89ÛÓ", "", tweet)

    tweet = re.sub(r"\x89ÛÏWhen", "When", tweet)

    tweet = re.sub(r"\x89ÛÏ", "", tweet)

    tweet = re.sub(r"China\x89Ûªs", "China's", tweet)

    tweet = re.sub(r"let\x89Ûªs", "let's", tweet)

    tweet = re.sub(r"\x89Û÷", "", tweet)

    tweet = re.sub(r"\x89Ûª", "", tweet)

    tweet = re.sub(r"\x89Û\x9d", "", tweet)

    tweet = re.sub(r"å_", "", tweet)

    tweet = re.sub(r"\x89Û¢", "", tweet)

    tweet = re.sub(r"\x89Û¢åÊ", "", tweet)

    tweet = re.sub(r"fromåÊwounds", "from wounds", tweet)

    tweet = re.sub(r"åÊ", "", tweet)

    tweet = re.sub(r"åÈ", "", tweet)

    tweet = re.sub(r"JapÌ_n", "Japan", tweet)    

    tweet = re.sub(r"Ì©", "e", tweet)

    tweet = re.sub(r"å¨", "", tweet)

    tweet = re.sub(r"SuruÌ¤", "Suruc", tweet)

    tweet = re.sub(r"åÇ", "", tweet)

    tweet = re.sub(r"å£3million", "3 million", tweet)

    tweet = re.sub(r"åÀ", "", tweet)

    

    # Contractions

    tweet = re.sub(r"he's", "he is", tweet)

    tweet = re.sub(r"there's", "there is", tweet)

    tweet = re.sub(r"We're", "We are", tweet)

    tweet = re.sub(r"That's", "That is", tweet)

    tweet = re.sub(r"won't", "will not", tweet)

    tweet = re.sub(r"they're", "they are", tweet)

    tweet = re.sub(r"Can't", "Cannot", tweet)

    tweet = re.sub(r"wasn't", "was not", tweet)

    tweet = re.sub(r"don\x89Ûªt", "do not", tweet)

    tweet = re.sub(r"aren't", "are not", tweet)

    tweet = re.sub(r"isn't", "is not", tweet)

    tweet = re.sub(r"What's", "What is", tweet)

    tweet = re.sub(r"haven't", "have not", tweet)

    tweet = re.sub(r"hasn't", "has not", tweet)

    tweet = re.sub(r"There's", "There is", tweet)

    tweet = re.sub(r"He's", "He is", tweet)

    tweet = re.sub(r"It's", "It is", tweet)

    tweet = re.sub(r"You're", "You are", tweet)

    tweet = re.sub(r"I'M", "I am", tweet)

    tweet = re.sub(r"shouldn't", "should not", tweet)

    tweet = re.sub(r"wouldn't", "would not", tweet)

    tweet = re.sub(r"i'm", "I am", tweet)

    tweet = re.sub(r"I\x89Ûªm", "I am", tweet)

    tweet = re.sub(r"I'm", "I am", tweet)

    tweet = re.sub(r"Isn't", "is not", tweet)

    tweet = re.sub(r"Here's", "Here is", tweet)

    tweet = re.sub(r"you've", "you have", tweet)

    tweet = re.sub(r"you\x89Ûªve", "you have", tweet)

    tweet = re.sub(r"we're", "we are", tweet)

    tweet = re.sub(r"what's", "what is", tweet)

    tweet = re.sub(r"couldn't", "could not", tweet)

    tweet = re.sub(r"we've", "we have", tweet)

    tweet = re.sub(r"it\x89Ûªs", "it is", tweet)

    tweet = re.sub(r"doesn\x89Ûªt", "does not", tweet)

    tweet = re.sub(r"It\x89Ûªs", "It is", tweet)

    tweet = re.sub(r"Here\x89Ûªs", "Here is", tweet)

    tweet = re.sub(r"who's", "who is", tweet)

    tweet = re.sub(r"I\x89Ûªve", "I have", tweet)

    tweet = re.sub(r"y'all", "you all", tweet)

    tweet = re.sub(r"can\x89Ûªt", "cannot", tweet)

    tweet = re.sub(r"would've", "would have", tweet)

    tweet = re.sub(r"it'll", "it will", tweet)

    tweet = re.sub(r"we'll", "we will", tweet)

    tweet = re.sub(r"wouldn\x89Ûªt", "would not", tweet)

    tweet = re.sub(r"We've", "We have", tweet)

    tweet = re.sub(r"he'll", "he will", tweet)

    tweet = re.sub(r"Y'all", "You all", tweet)

    tweet = re.sub(r"Weren't", "Were not", tweet)

    tweet = re.sub(r"Didn't", "Did not", tweet)

    tweet = re.sub(r"they'll", "they will", tweet)

    tweet = re.sub(r"they'd", "they would", tweet)

    tweet = re.sub(r"DON'T", "DO NOT", tweet)

    tweet = re.sub(r"That\x89Ûªs", "That is", tweet)

    tweet = re.sub(r"they've", "they have", tweet)

    tweet = re.sub(r"i'd", "I would", tweet)

    tweet = re.sub(r"should've", "should have", tweet)

    tweet = re.sub(r"You\x89Ûªre", "You are", tweet)

    tweet = re.sub(r"where's", "where is", tweet)

    tweet = re.sub(r"Don\x89Ûªt", "Do not", tweet)

    tweet = re.sub(r"we'd", "we would", tweet)

    tweet = re.sub(r"i'll", "I will", tweet)

    tweet = re.sub(r"weren't", "were not", tweet)

    tweet = re.sub(r"They're", "They are", tweet)

    tweet = re.sub(r"Can\x89Ûªt", "Cannot", tweet)

    tweet = re.sub(r"you\x89Ûªll", "you will", tweet)

    tweet = re.sub(r"I\x89Ûªd", "I would", tweet)

    tweet = re.sub(r"let's", "let us", tweet)

    tweet = re.sub(r"it's", "it is", tweet)

    tweet = re.sub(r"can't", "cannot", tweet)

    tweet = re.sub(r"don't", "do not", tweet)

    tweet = re.sub(r"you're", "you are", tweet)

    tweet = re.sub(r"i've", "I have", tweet)

    tweet = re.sub(r"that's", "that is", tweet)

    tweet = re.sub(r"i'll", "I will", tweet)

    tweet = re.sub(r"doesn't", "does not", tweet)

    tweet = re.sub(r"i'd", "I would", tweet)

    tweet = re.sub(r"didn't", "did not", tweet)

    tweet = re.sub(r"ain't", "am not", tweet)

    tweet = re.sub(r"you'll", "you will", tweet)

    tweet = re.sub(r"I've", "I have", tweet)

    tweet = re.sub(r"Don't", "do not", tweet)

    tweet = re.sub(r"I'll", "I will", tweet)

    tweet = re.sub(r"I'd", "I would", tweet)

    tweet = re.sub(r"Let's", "Let us", tweet)

    tweet = re.sub(r"you'd", "You would", tweet)

    tweet = re.sub(r"It's", "It is", tweet)

    tweet = re.sub(r"Ain't", "am not", tweet)

    tweet = re.sub(r"Haven't", "Have not", tweet)

    tweet = re.sub(r"Could've", "Could have", tweet)

    tweet = re.sub(r"youve", "you have", tweet)  

    tweet = re.sub(r"donå«t", "do not", tweet)   

            

    # Character entity references

    tweet = re.sub(r"&gt;", ">", tweet)

    tweet = re.sub(r"&lt;", "<", tweet)

    tweet = re.sub(r"&amp;", "&", tweet)

    

    # Typos, slang and informal abbreviations

    tweet = re.sub(r"w/e", "whatever", tweet)

    tweet = re.sub(r"w/", "with", tweet)

    tweet = re.sub(r"USAgov", "USA government", tweet)

    tweet = re.sub(r"recentlu", "recently", tweet)

    tweet = re.sub(r"Ph0tos", "Photos", tweet)

    tweet = re.sub(r"amirite", "am I right", tweet)

    tweet = re.sub(r"exp0sed", "exposed", tweet)

    tweet = re.sub(r"<3", "love", tweet)

    tweet = re.sub(r"amageddon", "armageddon", tweet)

    tweet = re.sub(r"Trfc", "Traffic", tweet)

    tweet = re.sub(r"8/5/2015", "2015-08-05", tweet)

    tweet = re.sub(r"WindStorm", "Wind Storm", tweet)

    tweet = re.sub(r"8/6/2015", "2015-08-06", tweet)

    tweet = re.sub(r"10:38PM", "10:38 PM", tweet)

    tweet = re.sub(r"10:30pm", "10:30 PM", tweet)

    tweet = re.sub(r"16yr", "16 year", tweet)

    tweet = re.sub(r"lmao", "laughing my ass off", tweet)   

    tweet = re.sub(r"TRAUMATISED", "traumatized", tweet)

    

    # Acronyms

    tweet = re.sub(r"MH370", "Malaysia Airlines Flight 370", tweet)

    tweet = re.sub(r"mÌ¼sica", "music", tweet)

    tweet = re.sub(r"okwx", "Oklahoma City Weather", tweet)

    tweet = re.sub(r"arwx", "Arkansas Weather", tweet)    

    tweet = re.sub(r"gawx", "Georgia Weather", tweet)  

    tweet = re.sub(r"scwx", "South Carolina Weather", tweet)  

    tweet = re.sub(r"cawx", "California Weather", tweet)

    tweet = re.sub(r"tnwx", "Tennessee Weather", tweet)

    tweet = re.sub(r"azwx", "Arizona Weather", tweet)  

    tweet = re.sub(r"alwx", "Alabama Weather", tweet)

    tweet = re.sub(r"wordpressdotcom", "wordpress", tweet)    

    tweet = re.sub(r"usNWSgov", "United States National Weather Service", tweet)

    tweet = re.sub(r"Suruc", "Sanliurfa", tweet)   

    

    # Grouping same words without embeddings

    tweet = re.sub(r"Bestnaijamade", "bestnaijamade", tweet)

    tweet = re.sub(r"SOUDELOR", "Soudelor", tweet)

    

    # remove stock market tickers like $GE

    tweet = re.sub(r'\$\w*', '', tweet)

    # remove old style retweet text "RT"

    tweet = re.sub(r'^RT[\s]+', '', tweet)

    # remove hyperlinks

    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)

    # remove hashtags

    # only removing the hash # sign from the word

    tweet = re.sub(r'#', '', tweet)

    # remove HTML tags

    tweet = re.sub(r'<.*?>', '', tweet)

    # remove words with digits (and emojis)

    tweet = re.sub('\w*\d\w*', '', tweet)

    # remove punctuations

    table = str.maketrans('', '', string.punctuation)

    tweet =  tweet.translate(table)

    

    return tweet
%%time

df['text'] = df['text'].apply(clean)

df_test['text'] = df_test['text'].apply(clean)
y_train = np.array(df['target']).tolist()
%%time

embedding_dim = 100

embedding_dict = {}



with open(f'../input/glove-global-vectors-for-word-representation/glove.twitter.27B.{embedding_dim}d.txt','r') as f:

    

    for line in f:

        

        values = line.split()

        word = values[0]

        vectors = np.asarray(values[1:],'float32')

        embedding_dict[word] = vectors

        

f.close()
print(f"There are {len(embedding_dict.keys())} tokens in this embedding representation")
stopwords = set(stopwords.words('english'))
%%time

def tokenize_corpus(df,  stopwords):

    

    corpus = []

    

    for tweet in df['text']:

        words = [word.lower() for word in word_tokenize(tweet) if ((word.isalpha() == 1) & (word not in stopwords))]

        corpus.append(words)

        

    return corpus
train_corpus = tokenize_corpus(df, stopwords)

len(train_corpus)
test_corpus = tokenize_corpus(df_test, stopwords)

len(test_corpus)
def build_vocab(tweets):

       

    vocab = {}

    

    for tweet in tweets:

        for word in tweet:

            try:

                vocab[word] += 1

            except KeyError:

                vocab[word] = 1                

    return vocab



def check_embeddings_coverage(X, embeddings):

    

    vocab = build_vocab(X)    

    

    covered = {}

    oov = {}    

    n_covered = 0

    n_oov = 0

    

    for word in vocab:

        try:

            covered[word] = embeddings[word]

            n_covered += vocab[word]

        except:

            oov[word] = vocab[word]

            n_oov += vocab[word]

            

    vocab_coverage = len(covered) / len(vocab)

    text_coverage = (n_covered / (n_covered + n_oov))

    

    sorted_oov = sorted(oov.items(), key=operator.itemgetter(1))[::-1]

    return sorted_oov, vocab_coverage, text_coverage
train_glove_oov, train_glove_vocab_coverage, train_glove_text_coverage = check_embeddings_coverage(train_corpus, embedding_dict)

test_glove_oov, test_glove_vocab_coverage, test_glove_text_coverage = check_embeddings_coverage(test_corpus, embedding_dict)



print('GloVe Embeddings cover {:.2%} of vocabulary and {:.2%} of text in Training Set'.format(train_glove_vocab_coverage, train_glove_text_coverage))

print('GloVe Embeddings cover {:.2%} of vocabulary and {:.2%} of text in Test Set'.format(test_glove_vocab_coverage, test_glove_text_coverage))
MAX_LEN = 50

tokenizer_obj = Tokenizer()

tokenizer_obj.fit_on_texts(train_corpus)

sequences = tokenizer_obj.texts_to_sequences(train_corpus)



tweet_pad = pad_sequences(sequences, maxlen=MAX_LEN, truncating='post', padding='post')
test_sequences = tokenizer_obj.texts_to_sequences(test_corpus)

test_pads = pad_sequences(test_sequences, maxlen=MAX_LEN, truncating='post', padding='post')
word_index = tokenizer_obj.word_index

print('Number of unique tokens:',len(word_index))
num_words = len(word_index) + 1 # +1 refer to pad token

embedding_matrix = np.zeros((num_words, embedding_dim))



for word,i in word_index.items():

    emb_vec = embedding_dict.get(word)

    if emb_vec is not None:

        embedding_matrix[i] = emb_vec
len(embedding_matrix), num_words
len(tweet_pad), len(y_train)
X, X_val, y, y_val = train_test_split(tweet_pad, y_train, test_size=0.2, random_state=42, stratify=y_train)



print('Length of train:', len(X))

print("Length of val:",   len(X_val))
X = np.asarray(X)

y = np.asarray(y)



X_val = np.asarray(X_val)

y_val = np.asarray(y_val)
np.unique(y_val, return_counts=True)
model=Sequential()



embedding=Embedding(num_words,

                    embedding_dim,

                    embeddings_initializer=Constant(embedding_matrix),

                    input_length=MAX_LEN,

                    trainable=False)



model.add(embedding)

model.add(SpatialDropout1D(0.2))

model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(32, activation='relu'))

model.add(Dense(16, activation='relu'))

model.add(Dense(1, activation='sigmoid'))





optimzer=Adam(learning_rate=1e-5)



model.compile(loss='binary_crossentropy', optimizer=optimzer, metrics=['accuracy'])

model.summary()
clr = CyclicLR(mode      = "triangular",

               base_lr   = 1e-5,

               max_lr    = 1e-4,

               step_size = 25)
early_stopping_callback = EarlyStopping(monitor='loss', mode='min', patience=5)
model_checkpoint_callback = ModelCheckpoint(

    filepath='best_model.h5',

    save_weights_only=True,

    monitor='val_accuracy',

    mode='max',

    save_best_only=True)
history = model.fit(X, y, 

                    batch_size=64, 

                    epochs=100, 

                    validation_data=(X_val, y_val), 

                    callbacks=[clr, early_stopping_callback, model_checkpoint_callback],

                    shuffle=True)
# summarize history for accuracy

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
#model.load_weights('best_model.h5')
def metrics(pred_tag, y_test):

    print("F1-score: ", f1_score(pred_tag, y_test))

    print("Precision: ", precision_score(pred_tag, y_test))

    print("Recall: ", recall_score(pred_tag, y_test))

    print("Acuracy: ", accuracy_score(pred_tag, y_test))

    print("-"*50)

    print(classification_report(pred_tag, y_test))
preds = model.predict_classes(X_val)



metrics(preds, y_val)
sample_sub = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')

perfect_sub = pd.read_csv('../input/perfect-submissiondisastertweetsnlp/perfect_submission.csv')
preds = model.predict_classes(test_pads)

preds = preds.astype(int).reshape(3263)

len(preds)
metrics(preds, perfect_sub['target'])
sub = pd.DataFrame({'id': sample_sub['id'].values.tolist(),'target': preds})

sub.head()
sub.to_csv('submission.csv',index=False)