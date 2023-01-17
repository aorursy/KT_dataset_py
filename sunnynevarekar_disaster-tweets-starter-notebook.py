# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test_df = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
train_df.head(10)
test_df.head(10)
print("Train set shape: {}".format(train_df.shape))

print("Test set shape: {}".format(test_df.shape))
import seaborn as sns

#Distribution of positive and negative examples in train set

pos_nev_ser = train_df['target'].value_counts()

sns.barplot(x=pos_nev_ser.index, y=pos_nev_ser).set_title("Negative and Postive samples")
from string import punctuation

from collections import defaultdict

import matplotlib.pyplot as plt



def show_punc_dist(df):

    pun_dict = defaultdict(int)



    for row in df['text']:

        for ch in row:

            if ch in punctuation:

                pun_dict[ch] += 1



    print(pun_dict) 



    x, y = zip(*pun_dict.items())



    plt.figure(figsize=(10, 5))

    plt.bar(x, y)

    plt.show()
show_punc_dist(train_df)
show_punc_dist(test_df)
import re



contractions = { 

"aren't": "are not",

"aren't": "are not",    

"can't": "cannot",

"cant": "cannot",    

"can't've": "cannot have",

"could've": "could have",

"couldn't": "could not",

"couldnt": "could not",    

"couldn't've": "could not have",

"didn't": "did not",

"didnt": "did not",    

"doesn't": "does not",

"doesnt": "does not",    

"don't": "do not",

"dont": "do not",

"hadn't": "had not",

"hadnt": "had not",    

"hadn't've": "had not have",

"hasn't": "has not",

"hasnt": "has not",    

"haven't": "have not",

"havent": "have not",    

"he'd": "he would",

"he'd've": "he would have",

"he'll": "he will",

"he'll've": "he will have",

"he's": "he is",

"hes": "he is",    

"how'd": "how did",

"how'd'y": "how do you",

"how'll": "how will",

"how's": "how is",

"i'd": "i would",

"i'd've": "i would have",

"i'll": "i will",

"i'll've": "i will have",

"i'm": "i am",

"i've": "i have",

"isn't": "is not",

"isnt": "is not",    

"it'd": "it would",

"it'd've": "it would have",

"it'll": "it will",

"it'll've": "it will have",

"it's": "it is",

"let's": "let us",

"lets": "let us",    

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

"shes": "she is",

"should've": "should have",

"shouldn't": "should not",

"shouldn't've": "should not have",

"so've": "so have",

"so's": "so is",

"that'd": "that would",

"that'd've": "that would have",

"that's": "that is",

"thats": "that is",    

"there'd": "there would",

"there'd've": "there would have",

"there's": "there is",

"theres": "there is",    

"they'd": "they would",

"they'd've": "they would have",

"they'll": "they will",

"theyll": "they will",    

"they'll've": "they will have",

"they're": "they are",

"theyre": "they are",    

"they've": "they have",

"to've": "to have",

"wasn't": "was not",

"wasnt": "was not",    

"we'd": "we would",

"we'd've": "we would have",

"we'll": "we will",

"we'll've": "we will have",

"we're": "we are",

"we've": "we have",

"weren't": "were not",

"what'll": "what will",

"what'll've": "what will have",

"what're": "what are",

"what's": "what is",

"whats" : "what is",    

"what've": "what have",

"when's": "when has / when is",

"when've": "when have",

"where'd": "where did",

"where's": "where is",

"wheres": "where is",    

"where've": "where have",

"who'll": "who will",

"who'll've": "who will have",

"who's": "who is",

"who've": "who have",

"why's": "why is",

"why've": "why have",

"will've": "will have",

"won't": "will not",

"wont": "will not",    

"won't've": "will not have",

"would've": "would have",

"wouldn't": "would not",

"wouldnt": "would not",    

"wouldn't've": "would not have",

"y'all": "you all",

"yall": "you all",    

"y'all'd": "you all would",

"y'all'd've": "you all would have",

"y'all're": "you all are",

"y'all've": "you all have",

"you'd": "you would",

"youd": "you would",    

"you'd've": "you would have",

"you'll": "you will",

"you'll've": "you will have",

"you're": "you are",

"youre": "you are",    

"you've": "you have"

}



def sub_contractions(contractions, text):

    tokens = [contractions[word] if word in contractions.keys() else word for word in text.split()]   

    return " ".join(tokens)



def clean_text(text):

    text = text.lower()

    text = re.sub("https?://\S+|www\.\S+", '', text)

    text = re.sub("<.*?>", '', text)

    text = re.sub("["u"\U0001F600-\U0001F64F"  # emoticons

                u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                u"\U0001F680-\U0001F6FF"  # transport & map symbols

                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                u"\U00002702-\U000027B0"

                u"\U000024C2-\U0001F251"

               "]+", '', text)

    

    text = sub_contractions(contractions, text)

    pun_str = "[{}]".format(punctuation).replace('@', '')

    text = re.sub(pun_str, '', text)

    text = re.sub("#", ' ', text)

    text = re.sub("\x89ûï", '', text)

    text = re.sub("\x89ûò", '', text)

    text = re.sub("\x89ûó", '', text)

    text = re.sub("\x89ûª", '', text)

    text = re.sub("\x89û", '', text)

    text = re.sub("\x9d", '', text)

    return text
train_df['text_cleaned'] = train_df['text'].apply(lambda x: clean_text(x))

test_df['text_cleaned'] = test_df['text'].apply(lambda x: clean_text(x))
from collections import defaultdict

from collections import Counter



def most_common_words(df, num_words=10):

    words = defaultdict(int)

    for row in df['text_cleaned']:

        for word in row.split():

            words[word] += 1

    return Counter(words).most_common(num_words)        
word, count = zip(*most_common_words(train_df))

plt.figure(figsize=(10, 5))

plt.bar(word, count)

plt.title("10 Most common words.")

plt.show()
word, count = zip(*most_common_words(train_df[train_df['target']==1]))

plt.figure(figsize=(10, 5))

plt.bar(word, count)

plt.title("10 Most common words in disaster tweets.")

plt.show()
word, count = zip(*most_common_words(train_df[train_df['target']==0]))

plt.figure(figsize=(10, 5))

plt.bar(word, count)

plt.title("10 Most common words in non-disaster tweets.")

plt.show()
train_df['word_length'] = train_df['text_cleaned'].str.split().apply(lambda x: len(x))

test_df['word_length'] = test_df['text_cleaned'].str.split().apply(lambda x:len(x))
sns.distplot(train_df['word_length']).set_title("Train samples word length distibution.")

plt.show()
sns.distplot(test_df['word_length']).set_title("Test samples word length distibution.")

plt.show()
embedding_dict={}

with open('../input/glove-global-vectors-for-word-representation/glove.6B.100d.txt','r') as f:

    for line in f:

        values=line.split()

        word=values[0]

        vectors=np.asarray(values[1:],'float32')

        embedding_dict[word]=vectors

        

print("Number of words in glove embedding: {}".format(len(embedding_dict)))        
oov_train = set()

for row in train_df['text_cleaned']:

    for word in row.split():

        if word not in embedding_dict:

            oov_train.add(word)



print("Out of vocabulary words in train set: {}".format(len(oov_train)))
print(oov_train)
oov_test = set()

for row in test_df['text_cleaned']:

    for word in row.split():

        if word not in embedding_dict:

            oov_test.add(word)



print("Out of vocabulary words in test set: {}".format(len(oov_test)))
print(oov_test)
from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences



tokenizer = Tokenizer()

#fit on whole text corpus

tokenizer.fit_on_texts(pd.concat([train_df['text_cleaned'], test_df['text_cleaned']]))



#train sequence

train_sentences = tokenizer.texts_to_sequences(train_df['text'])

train_sequences = pad_sequences(train_sentences, maxlen=50, padding='post', truncating='post')

print("Train sequence shape: {}".format(train_sequences.shape))



#test sequence

test_sentences = tokenizer.texts_to_sequences(test_df['text'])

test_sequences = pad_sequences(test_sentences, maxlen=50, padding='post', truncating='post')

print("Test sequence shape: {}".format(test_sequences.shape))
word_index = tokenizer.word_index

print("Number of unique words: {}".format(len(word_index)))
#dictionary to map index to word

index_to_word = dict()

for word, index in word_index.items():

    index_to_word[index] = word
from tqdm import tqdm

num_words = len(word_index) + 1

embedding_matrix = np.zeros((num_words, 100))



for word, i in tqdm(word_index.items()):

    if i > num_words:

        continue

    

    emb_vec=embedding_dict.get(word)

    if emb_vec is not None:

        embedding_matrix[i]=emb_vec    
print(embedding_matrix.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_sequences, train_df['target'], test_size=0.15, random_state=8)

print('Shape of train',X_train.shape)

print("Shape of Validation ",X_test.shape)
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Embedding, SpatialDropout1D, LSTM, Dense

from tensorflow.keras.initializers import Constant

from tensorflow.keras.optimizers import Adam
#define model

model = Sequential()

embedding = Embedding(num_words, 100, embeddings_initializer=Constant(embedding_matrix), input_length=50,trainable=False)

model.add(embedding)

model.add(SpatialDropout1D(0.2))

model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(1, activation='sigmoid'))



#adam optimizer

#optimzer=Adam(learning_rate=1e-5)



#compile model

model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])



model.summary()
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

earlyStopping = EarlyStopping(monitor='val_loss', patience=5)

chkpoint = ModelCheckpoint('best_weights.h5', monitor='val_loss', save_best_only=True)

history=model.fit(X_train,y_train, epochs=100, validation_data=(X_test,y_test), callbacks=[chkpoint, earlyStopping])
#load the saved best weights

model.load_weights('best_weights.h5')
from sklearn.metrics import f1_score

train_predictions = model.predict(train_sequences)[:, 0]

train_predictions = train_predictions > 0.5

train_predictions = train_predictions.astype(int)

print("Train set f1 score: {}".format(f1_score(train_df['target'], train_predictions)))
y_pred = model.predict(test_sequences)

y_pred = y_pred[:, 0]

y_pred = y_pred >0.5

y_pred = y_pred.astype(int)

sub=pd.DataFrame({'id':test_df['id'].values.tolist(),'target':y_pred})

sub.to_csv('submission.csv',index=False)