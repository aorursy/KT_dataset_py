# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import warnings

warnings.filterwarnings("ignore")

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

import seaborn as sns

import matplotlib.pyplot as plt

import nltk

from nltk.corpus import stopwords

from nltk import word_tokenize, sent_tokenize



import keras

from keras.preprocessing.text import Tokenizer, one_hot

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Dense, Embedding, LSTM, Flatten, SpatialDropout1D, Input, Bidirectional, Dropout

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback

from keras.regularizers import l2

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from sklearn.metrics import roc_auc_score

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#constants

eng_stopwords = set(stopwords.words("english"))

from nltk.stem.wordnet import WordNetLemmatizer 

lem = WordNetLemmatizer()





import os

print(os.listdir("../input"))

%matplotlib inline

# Any results you write to the current directory are saved as output.
#Importing datasets

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train_data = train['comment_text']

test_data = test['comment_text']

total_data = pd.DataFrame(train_data.append(test_data), columns = ['comment_text'])



print('Total_data: ', total_data.shape)

total_data.head()
y = train.iloc[:,2:]

y.head()
#checking for missing values

print("Missing values in total_data: ",total_data.isnull().sum())

print("Missing values in target variable: ",y.isnull().sum())
total_data['total_length'] = total_data['comment_text'].apply(len)

total_data['capitals'] = total_data['comment_text'].apply(lambda comment: sum(1 for c in comment if c.isupper()))

total_data['caps_vs_length'] = total_data.apply(lambda row: float(row['capitals'])/float(row['total_length']),

                                axis=1)

total_data['num_exclamation_marks'] = total_data['comment_text'].apply(lambda comment: comment.count('!'))

total_data['num_question_marks'] = total_data['comment_text'].apply(lambda comment: comment.count('?'))

total_data['num_punctuation'] = total_data['comment_text'].apply(

    lambda comment: sum(comment.count(w) for w in '.,;:'))

total_data['num_symbols'] = total_data['comment_text'].apply(

    lambda comment: sum(comment.count(w) for w in '*&$%'))

total_data['num_words'] = total_data['comment_text'].apply(lambda comment: len(comment.split()))

total_data['num_unique_words'] = total_data['comment_text'].apply(

    lambda comment: len(set(w for w in comment.split())))

total_data['words_vs_unique'] = total_data['num_unique_words'] / total_data['num_words']

total_data['num_smilies'] = total_data['comment_text'].apply(

    lambda comment: sum(comment.count(w) for w in (':-)', ':)', ';-)', ';)')))
import gensim

#to seperate sentenses into words

def preprocess(comment):

    """

    Function to build tokenized texts from input comment

    """

    return gensim.utils.simple_preprocess(comment, deacc=True, min_len=3)
#tokenize the comments

all_text = total_data.comment_text.apply(lambda x: preprocess(x))


def clean(word_list):

    """

    Function to clean the pre-processed word lists 

    

    Following transformations will be done

    1) Stop words removal from the nltk stopword list

    2) Lemmatization (Converting word to its root form : babies --> baby ; children --> child)

    """

    #remove stop words

    clean_words = [w for w in word_list if not w in eng_stopwords]



    #Lemmatize

    clean_words=[lem.lemmatize(word, "v") for word in clean_words]

    return(clean_words) 
#scale it to all text

all_text = all_text.apply(lambda x:clean(x))
#checking number of comments clean/unclean

print('Total number of comments in train data: ', train.shape[0])

print('Total number unclean comments train data: ', (y.sum()).sum())

#marking comments with no labels as clean

print('Total number clean comments train data: ', (train.shape[0]-(y.sum()).sum()))
#for clean comments we will make another column. We might need to remove this column in the end.

rowsums = y.sum(axis=1)

y['clean'] = (rowsums==0)
# Checking class imbalance

plt.figure(figsize=(8,4))

a = y.sum()

ax = sns.barplot(a.index, a.values, alpha=0.8)

plt.title('Class Balance')

plt.xlabel('Classes')

plt.ylabel('# of Occurrences', fontsize=12)

#adding the text labels

rects = ax.patches

labels = a.values

for rect, label in zip(rects, labels):

    height = rect.get_height()

    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')



plt.show()
y.shape[0]
#checking number of comments having multiple tags

k = rowsums.value_counts()

k
print('Number of comments with multiple tags: ', np.sum(k[2:7]))
temp = y.iloc[:,:-1]

corr = temp.corr()

plt.figure(figsize=(10,8))

sns.heatmap(corr,

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values, annot=True)
#checking proportion of different classes in train data

p = [i*100/a.sum() for i in a]



mapped = zip(a.index, p)

set(mapped)
y['clean'] = [1 if i is True else 0 for i in y['clean']]
m=1

plt.figure(figsize=(20,20))

for i in y.columns[:-1]:

    plt.subplot(3,3,m)

    subset=train[train[i]==1]

    text=subset.comment_text.values

    wc= WordCloud(background_color="black",max_words=2000)

    wc.generate(" ".join(text))

    plt.title(i, fontsize=50)

    plt.imshow(wc.recolor(colormap= 'summer'), alpha=0.98)

    m = m+1
'''

gensim is python library for training word embeddings in given data

for more information visit: 

1. https://machinelearningmastery.com/develop-word-embeddings-python-gensim/

2. http://kavita-ganesan.com/gensim-word2vec-tutorial-starter-code/#.XEoWKVwzbIV

'''

import gensim



embedding_vector_size = 100

# now training embeddings for each word 

model_1 = gensim.models.Word2Vec(sentences = all_text, size=embedding_vector_size, min_count=1, window=5, workers=4 )



# to get total number of unique words

words = list(model_1.wv.vocab)



print("vocab size:", len(words))
#len(sequence)

length = [len(x) for x in all_text]

plt.hist(length)

plt.xlabel('length of words')

plt.ylabel('frequency')
'''

Now we have trained the embeddings, we now have embedding vector for each word. We will

convert our text training data to numeric using theseword embeddings.

First, we need to make length of each input same, therefore we'll do padding. But padding happends 

on numeric data, therefore we'll convert texts to sequences using tokenize() function. Then add padding

Then we'll replace each non-zero numeric resulted from texts to sequences to its corresponding word

embedding.

'''

max_len = 130  

max_features = 6000

tokenizer = Tokenizer(num_words=max_features)       #keeps 6000 most common words

train_test_data = all_text                       # contains word tokens extracted from lines

tokenizer.fit_on_texts(train_test_data)

sequence = tokenizer.texts_to_sequences(train_test_data)

train_test_data = pad_sequences(sequence, maxlen = max_len)
'''

# Preparing embedding matrix

vocab_size = len(tokenizer.word_index)+1

embedding_matrix = np.zeros((vocab_size, embedding_vector_size))

# +1 is done because i starts from 1 instead of 0, and goes till len(vocab)

for  word, i in tokenizer.word_index.items():

    embedding_vector = model_1.wv[word]

    if embedding_vector is not None:

        embedding_matrix[i] = embedding_vector

'''
word_index = tokenizer.word_index

nb_words = min(max_features, len(word_index))

embedding_matrix = np.zeros((nb_words, embedding_vector_size))

for word, i in word_index.items():

    if i >= max_features: continue

    embedding_vector = model_1.wv[word]

    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
# Separating training and test data

ntrain = train.shape[0]

X = train_test_data[:ntrain,:]

test_data = train_test_data[ntrain:,:]

X = pd.DataFrame(X)

X = pd.concat([X,total_data.iloc[:ntrain,1:]],axis=1)

test_data = pd.DataFrame(test_data)

test_data = pd.concat([test_data,total_data.iloc[ntrain:,1:]],axis=1)
remove_n = 70000

y = y.reset_index()

drop_indices = np.random.choice(y['clean'].index, remove_n, replace=False)

#df_subset = X['clean'].drop(drop_indices)

X = X.reset_index()
print('Shape before dropping rows having clean labels: ', X.shape)

X=  X.drop(['index'],axis=1)

X = X.drop(drop_indices, axis=0)



print('Shape after dropping rows having clean labels: ', X.shape)
y1 = y.iloc[:,:-1]

y1 = y1.drop(drop_indices)

y1=  y1.drop(['index'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(X , y1, test_size=0.2, random_state=42, shuffle=True)
model = Sequential()



model.add(Embedding(input_dim = max_features, output_dim = embedding_vector_size, 

                    input_length = X.shape[1], weights = [embedding_matrix]))

model.add(Bidirectional(LSTM(64, dropout=0.25, recurrent_dropout=0.1, return_sequences=True)))

model.add(LSTM(32, return_sequences=True, dropout=0.25,))  # returns a sequence of vectors of dimension 32

model.add(LSTM(16, dropout=0.25,))  # return a single vector of dimension 32

model.add(Dense(10))

model.add(Dropout(0.3))

model.add(Dense(6, activation='sigmoid'))

model.compile(optimizer='RMSprop', loss='binary_crossentropy', metrics=['acc'])

print(model.summary())
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
class RocAucEvaluation(Callback):

    def __init__(self, validation_data=(), interval=1):

        super(Callback, self).__init__()



        self.interval = interval

        self.X_test, self.y_test = validation_data



    def on_epoch_end(self, epoch, logs={}):

        if epoch % self.interval == 0:

            y_pred = self.model.predict(self.X_test, verbose=0)

            score = roc_auc_score(self.y_test, y_pred)

            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))
RocAuc = RocAucEvaluation(validation_data=(X_test, y_test), interval=1)
history = model.fit(X_train, y_train, epochs = 6, batch_size = 1000, validation_data=(X_test, y_test),

                 callbacks=[RocAuc])
import matplotlib.pyplot as plt

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
y_pred = model.predict(test_data)

y_pred = pd.DataFrame(y_pred, columns = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"])
y_pred.to_csv('submission.csv', index=False)