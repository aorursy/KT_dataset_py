# Input data files are available in the "../input/" directory.
# For example, running this will list the files in the input directory
import os
print(os.listdir("../input"))
#Verify which embeddings are provided
!ls ../input/embeddings
#there are 4 embeddings provided with the dataset
#import packages
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import random
import spacy
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
import re
from bs4 import BeautifulSoup
import unicodedata
from collections import defaultdict
import string

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split

from nltk.corpus import stopwords
from sklearn.metrics import log_loss
from tqdm import tqdm
stopwords = stopwords.words('english')
sns.set_context('notebook')
#Code adapted from: https://www.kaggle.com/arunsankar/key-insights-from-quora-insincere-questions
#import the different datasets and print the characteristics of each
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sub = pd.read_csv('../input/sample_submission.csv')

#Print the different statistics of the different files
print('Train data: \nRows: {}\nCols: {}'.format(train.shape[0],train.shape[1]))
print(train.columns)

print('\nTest data: \nRows: {}\nCols: {}'.format(test.shape[0],test.shape[1]))
print(test.columns)

print('\nSubmission data: \nRows: {}\nCols: {}'.format(sub.shape[0],sub.shape[1]))
print(sub.columns)
#View the first 5 entries of the training data
train.head()
#View information about the train dataset
train.info()
##1306122 observations and 3 columns
#check for the number of positive and negative classes
pd.crosstab(index = train.target, columns = "count" )
#There seems to be unbalanced classes in the dataset [first issue]
#Code sourced from : https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-qiqc

#import the wordcloud package
from wordcloud import WordCloud, STOPWORDS

#Define the word cloud function with a max of 200 words
def plot_wordcloud(text, mask=None, max_words=200, max_font_size=100, figure_size=(24.0,16.0), 
                   title = None, title_size=40, image_color=False):
    stopwords = set(STOPWORDS)
    #define additional stop words that are not contained in the dictionary
    more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}
    stopwords = stopwords.union(more_stopwords)
    #Generate the word cloud
    wordcloud = WordCloud(background_color='black',
                    stopwords = stopwords,
                    max_words = max_words,
                    max_font_size = max_font_size, 
                    random_state = 42,
                    width=800, 
                    height=400,
                    mask = mask)
    wordcloud.generate(str(text))
    #set the plot parameters
    plt.figure(figsize=figure_size)
    if image_color:
        image_colors = ImageColorGenerator(mask);
        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");
        plt.title(title, fontdict={'size': title_size,  
                                  'verticalalignment': 'bottom'})
    else:
        plt.imshow(wordcloud);
        plt.title(title, fontdict={'size': title_size, 'color': 'black', 
                                  'verticalalignment': 'bottom'})
    plt.axis('off');
    plt.tight_layout()  
#Select insincere questions from training dataset
insincere = train.loc[train['target'] == 1]
#run the function on the insincere questions
plot_wordcloud(insincere["question_text"], title="Word Cloud of Insincere Questions")
#Select sincere questions from training dataset
sincere = train.loc[train['target'] == 0]
#run the function on the insincere questions
plot_wordcloud(sincere["question_text"], title="Word Cloud of Sincere Questions")
def ngram_extractor(text, n_gram):
    token = [token for token in text.lower().split(" ") if token != "" if token not in STOPWORDS]
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    return [" ".join(ngram) for ngram in ngrams]

# Function to generate a dataframe with n_gram and top max_row frequencies
def generate_ngrams(df, col, n_gram, max_row):
    temp_dict = defaultdict(int)
    for question in df[col]:
        for word in ngram_extractor(question, n_gram):
            temp_dict[word] += 1
    temp_df = pd.DataFrame(sorted(temp_dict.items(), key=lambda x: x[1])[::-1]).head(max_row)
    temp_df.columns = ["word", "wordcount"]
    return temp_df

#Function to construct side by side comparison plots
def comparison_plot(df_1,df_2,col_1,col_2, space):
    fig, ax = plt.subplots(1, 2, figsize=(20,10))
    
    sns.barplot(x=col_2, y=col_1, data=df_1, ax=ax[0], color="royalblue")
    sns.barplot(x=col_2, y=col_1, data=df_2, ax=ax[1], color="royalblue")

    ax[0].set_xlabel('Word count', size=14)
    ax[0].set_ylabel('Words', size=14)
    ax[0].set_title('Top words in sincere questions', size=18)

    ax[1].set_xlabel('Word count', size=14)
    ax[1].set_ylabel('Words', size=14)
    ax[1].set_title('Top words in insincere questions', size=18)

    fig.subplots_adjust(wspace=space)
    
    plt.show()
#Obtain sincere and insincere ngram based on 1 gram (top 20)
sincere_1gram = generate_ngrams(train[train["target"]==0], 'question_text', 1, 20)
insincere_1gram = generate_ngrams(train[train["target"]==1], 'question_text', 1, 20)
#compare the bar plots
comparison_plot(sincere_1gram,insincere_1gram,'word','wordcount', 0.25)
#Obtain sincere and insincere ngram based on 2 gram (top 20)
sincere_2gram = generate_ngrams(train[train["target"]==0], 'question_text', 2, 20)
insincere_2gram = generate_ngrams(train[train["target"]==1], 'question_text', 2, 20)
#compare the bar plots
comparison_plot(sincere_2gram,insincere_2gram,'word','wordcount', 0.25)
#Obtain sincere and insincere ngram based on 3 gram (top 20)
sincere_3gram = generate_ngrams(train[train["target"]==0], 'question_text', 3, 20)
insincere_3gram = generate_ngrams(train[train["target"]==1], 'question_text', 3, 20)
#compare the bar plots
comparison_plot(sincere_3gram,insincere_3gram,'word','wordcount', 0.25)
# Number of words in the questions
train["word_count"] = train["question_text"].apply(lambda x: len(str(x).split()))
test["word_count"] = test["question_text"].apply(lambda x: len(str(x).split()))

fig, ax = plt.subplots(figsize=(15,2))
sns.boxplot(x="word_count", y="target", data=train, ax=ax, palette=sns.color_palette("RdYlGn_r", 10), orient='h')
ax.set_xlabel('Word Count', size=10, color="#0D47A1")
ax.set_ylabel('Target', size=10, color="#0D47A1")
ax.set_title('[Horizontal Box Plot] Word Count distribution', size=12, color="#0D47A1")
plt.gca().xaxis.grid(True)
plt.show()
# Number of characters in the questions
train["char_length"] = train["question_text"].apply(lambda x: len(str(x)))
test["char_length"] = test["question_text"].apply(lambda x: len(str(x)))

fig, ax = plt.subplots(figsize=(15,2))
sns.boxplot(x="char_length", y="target", data=train, ax=ax, palette=sns.color_palette("RdYlGn_r", 10), orient='h')
ax.set_xlabel('Character Length', size=10, color="#0D47A1")
ax.set_ylabel('Target', size=10, color="#0D47A1")
ax.set_title('[Horizontal Box Plot] Character Length distribution', size=12, color="#0D47A1")
plt.gca().xaxis.grid(True)
plt.show()
# Number of stop words in the questions
train["stop_words_count"] = train["question_text"].apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))
test["stop_words_count"] = test["question_text"].apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))

fig, ax = plt.subplots(figsize=(15,2))
sns.boxplot(x="stop_words_count", y="target", data=train, ax=ax, palette=sns.color_palette("RdYlGn_r", 10), orient='h')
ax.set_xlabel('Number of stop words', size=10, color="#0D47A1")
ax.set_ylabel('Target', size=10, color="#0D47A1")
ax.set_title('[Horizontal Box Plot] Number of Stop Words distribution', size=12, color="#0D47A1")
plt.gca().xaxis.grid(True)
plt.show()
# Number of punctuations in the questions
train["punc_count"] = train["question_text"].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
test["punc_count"] = test["question_text"].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))

fig, ax = plt.subplots(figsize=(15,2))
sns.boxplot(x="punc_count", y="target", data=train[train['punc_count']<train['punc_count'].quantile(.99)], ax=ax, palette=sns.color_palette("RdYlGn_r", 10), orient='h')
ax.set_xlabel('Number of punctuations', size=10, color="#0D47A1")
ax.set_ylabel('Target', size=10, color="#0D47A1")
ax.set_title('[Horizontal Box Plot] Punctuation distribution', size=12, color="#0D47A1")
plt.gca().xaxis.grid(True)
plt.show()
# Number of upper case words in the questions
train["upper_words"] = train["question_text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
test["upper_words"] = test["question_text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

fig, ax = plt.subplots(figsize=(15,2))
sns.boxplot(x="upper_words", y="target", data=train[train['upper_words']<train['upper_words'].quantile(.99)], ax=ax, palette=sns.color_palette("RdYlGn_r", 10), orient='h')
ax.set_xlabel('Number of Upper case words', size=10, color="#0D47A1")
ax.set_ylabel('Target', size=10, color="#0D47A1")
ax.set_title('[Horizontal Box Plot] Upper case words distribution', size=12, color="#0D47A1")
plt.gca().xaxis.grid(True)
plt.show()
# Number of title words in the questions
train["title_words"] = train["question_text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
test["title_words"] = test["question_text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

fig, ax = plt.subplots(figsize=(15,2))
sns.boxplot(x="title_words", y="target", data=train[train['title_words']<train['title_words'].quantile(.99)], ax=ax, palette=sns.color_palette("RdYlGn_r", 10), orient='h')
ax.set_xlabel('Number of Title words', size=10, color="#0D47A1")
ax.set_ylabel('Target', size=10, color="#0D47A1")
ax.set_title('[Horizontal Box Plot] Title words distribution', size=12, color="#0D47A1")
plt.gca().xaxis.grid(True)
plt.show()
# Mean word length in the questions
train["word_length"] = train["question_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
test["word_length"] = test["question_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

fig, ax = plt.subplots(figsize=(15,2))
sns.boxplot(x="word_length", y="target", data=train[train['word_length']<train['word_length'].quantile(.99)], ax=ax, palette=sns.color_palette("RdYlGn_r", 10), orient='h')
ax.set_xlabel('Mean word length', size=10, color="#0D47A1")
ax.set_ylabel('Target', size=10, color="#0D47A1")
ax.set_title('[Horizontal Box Plot] Distribution of mean word length', size=12, color="#0D47A1")
plt.gca().xaxis.grid(True)
plt.show()
# nlp = spacy.load('en_core_web_sm')
# # Clean text before feeding it to model
# punctuations = string.punctuation

# # Define function to cleanup text by removing personal pronouns, stopwords, puncuation and reducing all characters to lowercase 
# def cleanup_text(docs, logging=False):
#     texts = []
#     for doc in tqdm(docs):
#         doc = nlp(doc, disable=['parser', 'ner'])
#         tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
#         #remove stopwords and punctuations
#         tokens = [tok for tok in tokens if tok not in stopwords and tok not in punctuations]
#         tokens = ' '.join(tokens)
#         texts.append(tokens)
#     return pd.Series(texts)
# # Cleanup text and make sure it retains original shape
# print('Original training data shape: ', train['question_text'].shape)
# train_cleaned = cleanup_text(train['question_text'], logging=True)
# print('Cleaned up training data shape: ', train_cleaned.shape)
#use 90-10 split for validation dataset
train, val_df = train_test_split(train, test_size=0.1)
# embdedding setup
# Source https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
#Based on https://www.kaggle.com/theoviel/improve-your-score-with-some-text-preprocessing
#GloVe is the most comprehensive word embedding

embeddings_index = {}
f = open('../input/embeddings/glove.840B.300d/glove.840B.300d.txt')
for line in tqdm(f):
    values = line.split(" ")
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))
# Convert values to embeddings
def text_to_array(text):
    empyt_emb = np.zeros(300)
    text = text[:-1].split()[:30]
    embeds = [embeddings_index.get(x, empyt_emb) for x in text]
    embeds+= [empyt_emb] * (30 - len(embeds))
    return np.array(embeds)

# train_vects = [text_to_array(X_text) for X_text in tqdm(train["question_text"])]
val_vects = np.array([text_to_array(X_text) for X_text in tqdm(val_df["question_text"][:3000])])
val_y = np.array(val_df["target"][:3000])
# Data providers
batch_size = 128

def batch_gen(train):
    n_batches = math.ceil(len(train) / batch_size)
    while True: 
        train = train.sample(frac=1.)  # Shuffle the data.
        for i in range(n_batches):
            texts = train.iloc[i*batch_size:(i+1)*batch_size, 1]
            text_arr = np.array([text_to_array(text) for text in texts])
            yield text_arr, np.array(train["target"][i*batch_size:(i+1)*batch_size])
#import Bi-Directional LSTM 
from keras.models import Sequential
from keras.layers import CuDNNLSTM, Dense, Bidirectional
import math
#Define the model architecture
model = Sequential()
model.add(Bidirectional(CuDNNLSTM(64, return_sequences=True),
                        input_shape=(30, 300)))
model.add(Bidirectional(CuDNNLSTM(64)))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
mg = batch_gen(train)
#remember to change the number of epochs
model.fit_generator(mg, epochs=10,
                    steps_per_epoch=1000,
                    validation_data=(val_vects, val_y),
                    verbose= True)
# prediction part
batch_size = 256
def batch_gen(test):
    n_batches = math.ceil(len(test) / batch_size)
    for i in range(n_batches):
        texts = test.iloc[i*batch_size:(i+1)*batch_size, 1]
        text_arr = np.array([text_to_array(text) for text in texts])
        yield text_arr

test = pd.read_csv("../input/test.csv")

all_preds = []
for x in tqdm(batch_gen(test)):
    all_preds.extend(model.predict(x).flatten())
#Submit predictions
y_te = (np.array(all_preds) > 0.5).astype(np.int)

submit_df = pd.DataFrame({"qid": test["qid"], "prediction": y_te})
submit_df.to_csv("submission.csv", index=False)