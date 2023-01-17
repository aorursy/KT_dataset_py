import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, Embedding, CuDNNLSTM
from keras.layers import Bidirectional, GlobalMaxPool1D, Dropout
from keras.models import Model

%load_ext autoreload
%autoreload 2
# Load in training and testing data
train_df = pd.read_csv('../input/train.csv')
train_df.head()
# Sentiment analysis requires the nltk package
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize the sentiment analyzer and define the sentiment extraction function
sid = SentimentIntensityAnalyzer()
def extract_sentiment(text):
    scores = sid.polarity_scores(text)
    return scores['compound']

train_df['sent'] = train_df['question_text'].apply(lambda x: extract_sentiment(x))
print('The average sentiment score for sincere questions is {:0.4f}'.\
          format(np.mean(train_df[train_df['target']==0]['sent'])))
print('The average sentiment score for insincere questions is {:0.4f}'. \
          format(np.mean(train_df[train_df['target']==1]['sent'])))
sent_sinc = train_df[train_df['target']==0]['sent'].values
sent_insinc = train_df[train_df['target']==1]['sent'].values

sns.kdeplot(sent_sinc, shade=1, color="skyblue", label="Sincere")
sns.kdeplot(sent_insinc, shade=1, color="red", label="Insincere")

# Plot formatting
plt.legend(prop={'size': 12}, title = 'Sincerity')
plt.title('Sentiment Density Plot')
plt.xlabel('Sentiment')
plt.ylabel('Density')
train_df['word_count'] = train_df['question_text'].apply(lambda x: len(x.split()))
print('The average word length of sincere questions in the training set is {0:.1f}.'\
          .format(np.mean(train_df[train_df['target']==0]['word_count'])))
print('The average word length of insincere questions in the training set is {0:.1f}.' \
          .format(np.mean(train_df[train_df['target']==1]['word_count'])))
print('The maximum word length for a question in the training set is {0:.0f}.'\
          .format(np.max(train_df['word_count'])))
wc_sinc = train_df[train_df['target']==0]['word_count'].values
wc_insinc = train_df[train_df['target']==1]['word_count'].values

sns.kdeplot(wc_sinc, shade=1, color="skyblue", label="Sincere")
sns.kdeplot(wc_insinc, shade=1, color="red", label="Insincere")

# Plot formatting
plt.legend(prop={'size': 12}, title = 'Sincerity')
plt.title('Word Count Density Plot')
plt.xlabel('Word Count')
plt.ylabel('Density')
from wordcloud import WordCloud,STOPWORDS

def wordcloud_draw(data, title, color='black'):
    words = ' '.join(data)
    wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color=color,
                      width=2500,
                      height=2000
                     ).generate(words)
    plt.figure(1,figsize=(13, 13))
    plt.imshow(wordcloud)
    plt.title(title, fontdict={'fontsize':18})
    plt.axis('off')
    plt.show()

title = "Sincere words"
wordcloud_draw(train_df[train_df['target']==0]['question_text'], title, 'white')
title = "Insincere Words"
wordcloud_draw(train_df[train_df['target']==1]['question_text'], title, 'black')
# Extract the training data and corresponding labels
text = train_df['question_text'].fillna('unk').values
labels = train_df['target'].values

# Split into training and validation sets by making use of the scikit-learn
# function train_test_split
X_train, X_val, y_train, y_val = train_test_split(text, labels,\
                                                  test_size=0.2)
embed_size = 300 # Size of each word vector
max_words = 50000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 100 # max number of words in a question to use
## Tokenize the sentences
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(list(X_train))

# The tokenizer will assign an integer value to each word in the dictionary
# and then convert each string of words into a list of integer values
X_train = tokenizer.texts_to_sequences(X_train)
X_val = tokenizer.texts_to_sequences(X_val)

word_index = tokenizer.word_index
print('The word index consists of {} unique tokens.'.format(len(word_index)))

## Pad the sentences 
X_train = pad_sequences(X_train, maxlen=maxlen)
X_val = pad_sequences(X_val, maxlen=maxlen)
# Create the embedding dictionary from the word embedding file
embedding_dict = {}
filename = os.path.join('../input/embeddings/', 'glove.840B.300d/glove.840B.300d.txt')
with open(filename) as f:
    for line in f:
        line = line.split()
        token = line[0]
        try:
            coefs = np.asarray(line[1:], dtype='float32')
            embedding_dict[token] = coefs
        except:
            pass
print('The embedding dictionary has {} items'.format(len(embedding_dict)))
# Create the embedding layer weight matrix
embed_mat = np.zeros(shape=[max_words, embed_size])
for word, idx in word_index.items():
    # Word index is ordered from most frequent to least frequent
    # Ignore words that occur less frequently
    if idx >= max_words: continue
    vector = embedding_dict.get(word)
    if vector is not None:
        embed_mat[idx] = vector
def create_lstm():
    input = Input(shape=(maxlen,))
    
    # Embedding layer has fixed weights, so set 'trainable' to False
    x = Embedding(max_words, embed_size, weights=[embed_mat], trainable=False)(input)
    x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(16, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=input, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    
    return model
# Create and train network
lstm = create_lstm()
lstm.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=3, batch_size=512)
test_df = pd.read_csv('../input/test.csv')
X_test = test_df['question_text'].values

# Perform the same preprocessing as was done on the training set
X_test = tokenizer.texts_to_sequences(X_test)
X_test = pad_sequences(X_test, maxlen=maxlen)

# Make predictions, ensure that predictions are in integer form
preds = np.rint(lstm.predict([X_test], batch_size=1024, verbose=1)).astype('int')
test_df['prediction'] = preds
n=5
sin_sample = test_df.loc[test_df['prediction'] == 0]['question_text'].head(n)
print('Sincere Samples:')
for idx, row in enumerate(sin_sample):
    print('{}'.format(idx+1), row)

print('\n')
print('Insincere Samples:')
insin_sample = test_df.loc[test_df['prediction'] == 1]['question_text'].head(n)
for idx, row in enumerate(insin_sample):
    print('{}'.format(idx+1), row)
test_df = test_df.drop('question_text', axis=1)
test_df.to_csv('submission.csv', index=False)