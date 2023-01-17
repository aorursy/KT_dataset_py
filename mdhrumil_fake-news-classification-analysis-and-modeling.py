# Importing the libraries.

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import nltk
from wordcloud import WordCloud
import textstat as ts
from numpy.random import seed
seed(100)


nltk.download('stopwords')


%matplotlib inline
# Importing the dataset.

data_path = "../input/fake-and-real-news-dataset/"

true_news_data = pd.read_csv(data_path + "True.csv")
fake_news_data = pd.read_csv(data_path + "Fake.csv")
"""
A utility function to calculate the avg. length of the text (in number of words.)

"""

def avg_text_length(dataframe):
    rowCounts = [len(x.split()) for x in dataframe["text"].tolist()]
    avgCount = sum(rowCounts)//len(rowCounts)
    
    return avgCount
print("Avg. text length for true news: {}".format(avg_text_length(true_news_data)))
print("Avg. text length for fake news: {}".format(avg_text_length(fake_news_data)))
# Plotting the histogram for meaningful insights.

true_lengths = pd.Series([len(x.split()) for x in true_news_data["text"].tolist()])
fake_lengths = pd.Series([len(x.split()) for x in fake_news_data["text"].tolist()])

plt.figure(figsize = (15,10))
plt.hist(true_lengths, bins = 100, range = [0, 2000], color = "brown", label = "true-news length")
plt.hist(fake_lengths, bins = 100, range = [0, 2000], color = "blue", alpha = 0.5, label = "fake-news length")
plt.xlabel("Length of a news article in no. of words.", fontsize = 15)
plt.ylabel("Number of articles with x word length.", fontsize = 15)
plt.title("Class wise distribution of length of news article.", fontsize = 15)
plt.legend()
plt.show()
"""
A utility function to calculate the five number summary of the length of the text of news articles.

"""

def five_num_summary(dataframe):
    quartiles = np.percentile([len(x.split()) for x in dataframe["text"].tolist()], [0, 25, 50, 75, 100])
    
    return quartiles
true_summ = five_num_summary(true_news_data)

print("Text length - five number summary for true news data:")
for i in range(0, 101,25):
    print("{}'th %ile: {}".format(i, true_summ[i//25]))
    
    
print("\n")


fake_summ = five_num_summary(fake_news_data)

print("Text length - five number summary for fake news data:")
for i in range(0, 101,25):
    print("{}'th %ile: {}".format(i, fake_summ[i//25]))
stopwords = nltk.corpus.stopwords.words("english")

"""
A utility function to calculate the average % of stopwords content in true and fake news.

"""

def stopwords_frequency(dataframe):
    words_list = [x.split() for x in dataframe["text"].tolist()]
    frequencies = []
    for row in words_list:
        if(len(row)) > 0:
            row_frequency = len([w for w in row if w in stopwords])
            row_frequency = (row_frequency/len(row))*100

            frequencies.append(row_frequency)
    
    avg = sum(frequencies)/len(frequencies)
    
    return avg
true_frequencies = stopwords_frequency(true_news_data)
fake_frequencies = stopwords_frequency(fake_news_data)

print("Average stopwords frequency in true-news: {}".format(true_frequencies))
print("Average stopwords frequency in fake-news: {}".format(fake_frequencies))
cloud = WordCloud(width=1440, height=1080).generate(" ".join(true_news_data["text"].astype(str)))
plt.figure(figsize=(15, 10))
plt.imshow(cloud)
plt.axis('off')

print("True news articles wordcloud.")
cloud = WordCloud(width=1440, height=1080).generate(" ".join(fake_news_data["text"].astype(str)))
plt.figure(figsize=(15, 10))
plt.imshow(cloud)
plt.axis('off')

print("Fake news articles wordcloud.")
true_news_readability = []
fake_news_readability = []

for sentence in true_news_data["text"].tolist():
    temp = ts.flesch_reading_ease(sentence)
    true_news_readability.append(temp)
    
for sentence in fake_news_data["text"].tolist():
    temp = ts.flesch_reading_ease(sentence)
    fake_news_readability.append(temp)
true_readability_df = pd.Series(true_news_readability)
fake_readability_df = pd.Series(fake_news_readability)

plt.figure(figsize = (15,10))
plt.hist(true_readability_df, bins = 10, range = [0, 100], color = "brown", label = "true-news readability")
plt.hist(fake_readability_df, bins = 10, range = [0, 100], color = "blue", alpha = 0.5, label = "fake-news readability")
plt.xlabel("Flesch readability easiness for a news article.", fontsize = 15)
plt.ylabel("Number of articles with x word length.", fontsize = 15)
plt.title("Class wise Flesch readability ease score distribution.", fontsize = 15)
plt.legend()
plt.show()
fake_datewise_counts = fake_news_data.groupby('date').date.agg([('count', 'count')]).reset_index().sort_values(by = "count", ascending = False)

fake_datewise_counts = fake_datewise_counts.head(50)
plt.figure(figsize = (15,10))
plt.xticks(rotation = 90)
plt.bar(fake_datewise_counts["date"], fake_datewise_counts["count"], align = "center", color = "orange")
plt.xlabel("Date.", fontsize = 15)
plt.ylabel("Count of fake news articles released.", fontsize = 15)
plt.title("Date-wise distribution of number of fake news articles released.", fontsize = 15)
plt.show()
fake_news_data.loc[fake_news_data["date"] == "May 10, 2017"]["title"].tolist()[:10]
# Introducing the label column stating if the news article is True or False.

true_news_data["label"] = "True"
fake_news_data["label"] = "Fake"

all_news_data = true_news_data.append(fake_news_data, ignore_index = True)
del all_news_data['title']
del all_news_data['subject']
del all_news_data['date']

all_news_data.head()
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import re


# Preprocessing function definitions


"""
A utility function to remove punctuations from the text.

"""

def remove_punctuations(text):
    text  = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    return text


"""
A utility function to remove numerical characters from the text.

"""

def remove_nums(text):
    text = ''.join([i for i in text if not i.isdigit()])         
    return text


"""
A utility function to remove URL links from the text.

"""

def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)


"""
A utility function to remove HTML tags from the text.

"""

def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)


"""
A utility function to remove emojis from the text.

"""

def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


stop_words = set(stopwords.words('english'))

"""
A utility function to remove stopwords from the text.

"""

def clean_stopwords(text):
    res = [w for w in text.split() if not w in stop_words]
    res_string = " ".join(str(x) for x in res)
    return res_string

all_news_data_processed = all_news_data.copy()

all_news_data_processed["text"] = all_news_data_processed["text"].apply(lambda x: remove_punctuations(x))
all_news_data_processed["text"] = all_news_data_processed["text"].apply(lambda x: remove_nums(x))
all_news_data_processed["text"] = all_news_data_processed["text"].apply(lambda x: remove_URL(x))
all_news_data_processed["text"] = all_news_data_processed["text"].apply(lambda x: remove_html(x))
all_news_data_processed["text"] = all_news_data_processed["text"].apply(lambda x: remove_emoji(x))
all_news_data_processed["text"] = all_news_data_processed["text"].apply(lambda x: clean_stopwords(x))

# Shuffling the rows.

all_news_data_processed = all_news_data_processed.sample(frac = 1).reset_index(drop=True).reset_index(drop = True)

all_news_data_processed.head(10)
all_news_data_processed = all_news_data_processed[all_news_data_processed["text"].str.split().str.len().gt(0)]
len(all_news_data_processed)
train_X = all_news_data_processed.loc[:38000, "text"].values
train_Y = all_news_data_processed.loc[:38000, "label"].values
validation_X = all_news_data_processed.loc[38000:, "text"].values
validation_Y = all_news_data_processed.loc[38000:, "label"].values
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(train_X)
validation_vectors = vectorizer.transform(validation_X)
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import time

start = time.time()

mnb_classifier = MultinomialNB().fit(train_vectors, train_Y)
linear_svc_classifier = SVC(kernel = "linear").fit(train_vectors, train_Y)

end = time.time()

print("Trained 2 models in {} seconds.".format(end - start))
from  sklearn.metrics  import accuracy_score

mnb_predicted = mnb_classifier.predict(validation_vectors)
linear_svc_predicted = linear_svc_classifier.predict(validation_vectors)


print("Validation accuracy - Multinomial Naive Bayes: {}".format(accuracy_score(validation_Y, mnb_predicted)))
print("Validation accuracy - Linear Support Vector Classifier: {}".format(accuracy_score(validation_Y, linear_svc_predicted)))
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
print("A detailed report on the model performance:")

print("Model type: Multinomial Naive Bayes")
print(classification_report(validation_Y, mnb_predicted))

print("\n")

print("Model type: Support Vector Machines")
print(classification_report(validation_Y, linear_svc_predicted))
all_news_data_processed['label'] = all_news_data_processed['label'].map( {'Fake':1, 'True':0} )

all_news_data_processed.head(10)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


tf.random.set_seed(100)


vocab_size = 1000000
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size = 38000
max_length = 100

# Splitting the train and the test sentences list.

temp = [x for x in all_news_data_processed["text"].tolist()]
train_sentences = temp[:training_size]
test_sentences = temp[training_size:]

# Splitting the train and the test labels list.

temp2 = [x for x in all_news_data_processed["label"].tolist()]
train_labels = temp2[:training_size]
test_labels = temp2[training_size:]


tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_sentences)

word_index = tokenizer.word_index

train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
# Conversion to numpy array equivalents

train_padded = np.array(train_padded)
train_labels = np.array(train_labels)

test_padded = np.array(test_padded)
test_labels = np.array(test_labels)
"""
A utility function plot learning curves for the trained model.

"""

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()
# Regular dense neural network with word-embeddings dimension = 5.

embedding_dim = 5

dnn_model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
dnn_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
num_epochs = 10

early_stopping_callback_loss = tf.keras.callbacks.EarlyStopping(monitor = 'loss', patience = 1)
early_stopping_callback_val_loss = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 1)

dnn_history = dnn_model.fit(train_padded, train_labels, epochs=num_epochs, validation_split = 0.2, \
                        callbacks = [early_stopping_callback_loss, early_stopping_callback_val_loss], verbose=1)

plot_graphs(dnn_history, "accuracy")
plot_graphs(dnn_history, "loss")
dnn_results = dnn_model.evaluate(test_padded, test_labels, batch_size=128)

print('Regular dense network - test loss: {}'.format(dnn_results[0]))
print('Regular dense network - test accuracy: {}'.format(dnn_results[1]))
dnn_pred = (dnn_model.predict(test_padded) >= 0.5).astype("int")

print("A detailed report on the model performance:")

print("Model type: Regular dense neural network.")
print(classification_report(test_labels, dnn_pred))
print(confusion_matrix(test_labels, dnn_pred))
# Recurrent LSTM network with word-embeddings dimension = 10

embedding_dim = 10

lstm_model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.LSTM(12),
    tf.keras.layers.Dense(12, activation = "relu"),
    tf.keras.layers.Dense(1, activation = "sigmoid")
])

lstm_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
num_epochs = 10

early_stopping_callback_loss = tf.keras.callbacks.EarlyStopping(monitor = 'loss', patience = 1)
early_stopping_callback_val_loss = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 1)

lstm_history = lstm_model.fit(train_padded, train_labels, epochs=num_epochs, validation_split = 0.2, \
                         callbacks = [early_stopping_callback_loss, early_stopping_callback_val_loss], verbose=1)

plot_graphs(lstm_history, "accuracy")
plot_graphs(lstm_history, "loss")
# Recurrent LSTM network with word-embeddings dimension = 10

embedding_dim = 10

lstm_model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.LSTM(12),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(18, activation = "relu"),
    tf.keras.layers.Dense(1, activation = "sigmoid")
])

lstm_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
num_epochs = 10

early_stopping_callback_loss = tf.keras.callbacks.EarlyStopping(monitor = 'loss', patience = 1)
early_stopping_callback_val_loss = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 1)

lstm_history = lstm_model.fit(train_padded, train_labels, epochs=num_epochs, validation_split = 0.2, \
                         callbacks = [early_stopping_callback_loss, early_stopping_callback_val_loss], verbose=1)

plot_graphs(lstm_history, "accuracy")
plot_graphs(lstm_history, "loss")
lstm_results = lstm_model.evaluate(test_padded, test_labels, batch_size=128)

print('LSTM Model - test loss: {}'.format(lstm_results[0]))
print('LSTM Model - test accuracy: {}'.format(lstm_results[1]))
lstm_pred = (lstm_model.predict(test_padded) >= 0.5).astype("int")

print("A detailed report on the model performance:")

print("Model type: LSTM recurrent network.")
print(classification_report(test_labels, lstm_pred))
print(confusion_matrix(test_labels, lstm_pred))
all_news_data_title = true_news_data.append(fake_news_data, ignore_index = True)

all_news_data_processed_title = all_news_data_title.copy()

all_news_data_processed_title["title"] = all_news_data_processed_title["title"].apply(lambda x: remove_punctuations(x))
all_news_data_processed_title["title"] = all_news_data_processed_title["title"].apply(lambda x: remove_nums(x))
all_news_data_processed_title["title"] = all_news_data_processed_title["title"].apply(lambda x: remove_URL(x))
all_news_data_processed_title["title"] = all_news_data_processed_title["title"].apply(lambda x: remove_html(x))
all_news_data_processed_title["title"] = all_news_data_processed_title["title"].apply(lambda x: remove_emoji(x))
all_news_data_processed_title["title"] = all_news_data_processed_title["title"].apply(lambda x: clean_stopwords(x))

all_news_data_processed_title['label'] = all_news_data_processed_title['label'].map( {'Fake':1, 'True':0} )

del all_news_data_processed_title["text"]
del all_news_data_processed_title["subject"]
del all_news_data_processed_title["date"]

all_news_data_processed_title = all_news_data_processed_title[all_news_data_processed_title["title"].str.split().str.len().gt(0)]

all_news_data_processed_title = all_news_data_processed_title.sample(frac = 1).reset_index(drop=True).reset_index(drop = True)

all_news_data_processed_title.head(10)
tf.random.set_seed(100)

temp = [x for x in all_news_data_processed_title["title"].tolist()]
train_sentences_title = temp[:training_size]
test_sentences_title = temp[training_size:]

temp2 = [x for x in all_news_data_processed_title["label"].tolist()]
train_labels_title = temp2[:training_size]
test_labels_title = temp2[training_size:]


tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_sentences)

word_index = tokenizer.word_index

train_sequences_title = tokenizer.texts_to_sequences(train_sentences_title)
train_padded_title = pad_sequences(train_sequences_title, maxlen=max_length, padding=padding_type, truncating=trunc_type)

test_sequences_title = tokenizer.texts_to_sequences(test_sentences_title)
test_padded_title = pad_sequences(test_sequences_title, maxlen=max_length, padding=padding_type, truncating=trunc_type)


train_padded_title = np.array(train_padded_title)
train_labels_title = np.array(train_labels_title)

test_padded_title = np.array(test_padded_title)
test_labels_title = np.array(test_labels_title)
# Regular dense neural network with word-embeddings dimension = 5.

embedding_dim = 5

dnn_model_title = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
dnn_model_title.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

num_epochs_title = 10

early_stopping_callback_loss = tf.keras.callbacks.EarlyStopping(monitor = 'loss', patience = 1)
early_stopping_callback_val_loss = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 1)

dnn_history_title = dnn_model_title.fit(train_padded_title, train_labels_title, epochs=num_epochs_title, validation_split=0.2,
                        callbacks = [early_stopping_callback_loss, early_stopping_callback_val_loss], verbose=1)

plot_graphs(dnn_history_title, "accuracy")
plot_graphs(dnn_history_title, "loss")
dnn_results_title = dnn_model_title.evaluate(test_padded_title, test_labels_title, batch_size=128)

print('Regular dense network - test loss: {}'.format(dnn_results_title[0]))
print('Regular dense network - test accuracy: {}'.format(dnn_results_title[1]))
dnn_pred_title = (dnn_model_title.predict(test_padded_title) >= 0.5).astype("int")

print("A detailed report on the model performance:")

print("Model type: Regular dense neural network.")
print(classification_report(test_labels_title, dnn_pred_title))
print(confusion_matrix(test_labels_title, dnn_pred_title))