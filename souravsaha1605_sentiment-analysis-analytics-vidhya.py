import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import numpy as np
train = pd.read_csv("../datasets/analytics_vidhya/sentiment_analysis/train_hackathon_sentiment_analysis_1.csv")
test = pd.read_csv("../datasets/analytics_vidhya/sentiment_analysis/test_hackathon_sentiment_analysis_1.csv")

train.head()
test.head()
train["length"] = train["tweet"].map(lambda x : len(x))
train["length"].hist()
import string
def message_cleaning(message):
    Test_punc_removed = [char for char in message if char not in string.punctuation]
    Test_punc_removed_join = ''.join(Test_punc_removed)
    sentence = ''.join(i if ord(i)<128 else ' ' for i in Test_punc_removed_join)
#     sentence = re.sub(r'[0-9]+', '', sentence)
    Test_punc_removed_join_clean = [word for word in sentence.split()]
    return Test_punc_removed_join_clean
train["tweet"] = train["tweet"].map(lambda x : message_cleaning(x))
test["tweet"] = test["tweet"].map(lambda x : message_cleaning(x))
from sklearn.feature_extraction.text import CountVectorizer
# Define the cleaning pipeline we defined earlier
vectorizer = CountVectorizer(analyzer = message_cleaning, min_df=5)
train_countvectorizer = vectorizer.fit_transform(train['tweet'])
test_countvectorizer = vectorizer.fit_transform(test["tweet"])
def show_len():
    print (len(x_train[0]))
    print (len(x_train[1]))
# Tokenize our training data
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

num_words = 10000
oov_token = '<UNK>'
pad_type = 'post'
trunc_type = 'post'
tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
tokenizer.fit_on_texts(train['tweet'])

# Get our training data word index
word_index = tokenizer.word_index

# Encode training data sentences into sequences
x_train = tokenizer.texts_to_sequences(train['tweet'])
x_test = tokenizer.texts_to_sequences(test["tweet"])
show_len()
# print (train_sequences)

# Get max training sequence length
maxlen = max([len(x) for x in train_sequences])
print (maxlen)
x_train = pad_sequences(x_train, value = 1, padding=pad_type,  maxlen=256)
x_test = pad_sequences(x_test, value = 1, padding=pad_type, maxlen=256)
show_len()
word_index["the"]
word_index
y_train = train["label"]
y_train
print (x_train[1])
print (x_train[2])
print (x_train[0])
train_countvectorizer
train["length"].max()
x_train = train["tweet"]
y_train = train["label"]

x_test = test["tweet"]
x_train[0]

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Embedding, GlobalAveragePooling1D

model = Sequential([
    Embedding(10000, 16),
    GlobalAveragePooling1D(),
    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid") #here sigmoid is used since it's a binary classification problem, and it's the output layer
])

model.compile(
    loss = "binary_crossentropy",
    optimizer = "adam",
    metrics = ["acc"]
)

model.summary()
#we don't want to display all the information in the notebook, i.e. the verbose instead we can show simple information to 
#understand that the training is on progress
from tensorflow.python.keras.callbacks import LambdaCallback

simple_logging = LambdaCallback(on_epoch_end = lambda e, l: print(e, end='.'))

E = 20

h = model.fit(
    x_train, y_train,
    validation_split = 0.2,
    epochs = E,
    callbacks = [simple_logging],
    verbose = False
)
import matplotlib.pyplot as plt
%matplotlib inline

plt.plot(range(E), h.history["acc"], label="Training")
plt.plot(range(E), h.history["val_acc"], label="Validation")
plt.legend()
plt.show()
#we don't want to display all the information in the notebook, i.e. the verbose instead we can show simple information to 
#understand that the training is on progress
from tensorflow.python.keras.callbacks import LambdaCallback

simple_logging = LambdaCallback(on_epoch_end = lambda e, l: print(e, end='.'))

E = 13

model_13 = model.fit(
    x_train, y_train,
    validation_split = 0.2,
    epochs = E,
    callbacks = [simple_logging],
    verbose = False
)
import matplotlib.pyplot as plt
%matplotlib inline

plt.plot(range(E), model_13.history["acc"], label="Training")
plt.plot(range(E), model_13.history["val_acc"], label="Validation")
plt.legend()
plt.show()
prediction = model.predict(x_test)
prediction_np = np.around(prediction)
sentiment_predicted_labels = pd.DataFrame({"id":test["id"], 'label': prediction_np[:, 0]})
sentiment_predicted_labels["label"] = sentiment_predicted_labels["label"].astype(int)
sentiment_predicted_labels

sentiment_predicted_labels.to_csv("../datasets/analytics_vidhya/sentiment_analysis/submission_twitter.csv", index=False)
train_tweets_df = pd.read_csv("../datasets/analytics_vidhya/sentiment_analysis/train_hackathon_sentiment_analysis_1.csv")
test_tweets_df = pd.read_csv("../datasets/analytics_vidhya/sentiment_analysis/test_hackathon_sentiment_analysis_1.csv")
train_tweets_df.drop("id", axis=1, inplace=True)
import spacy
nlp = spacy.load('en_core_web_sm')

def remove_stop_words_punctuation(str_value):
    doc = nlp(str_value)
    word_list = []
    for token in doc:
        if ((token.is_stop != True) &(token.pos_ != "PUNCT")):
            token = str(token)
            token = ''.join(i if ord(i)<128 else ' ' for i in token)
            word_list.append(token.strip())
    return word_list 
train_tweets_df.loc[0 : "tweet"]
remove_stop_words_punctuation("ate @user isz that youuu?ðððððð...")
message_cleaning("ate @user isz that youuu?ðððððð...")
from sklearn.feature_extraction.text import CountVectorizer
# Define the cleaning pipeline we defined earlier
vectorizer = CountVectorizer(analyzer = message_cleaning, min_df=5)
vectorizer.fit(train_tweets_df['tweet'])
train_tweets_countvectorizer = vectorizer.transform(train_tweets_df['tweet'])
test_tweets_countvectorizer = vectorizer.transform(test_tweets_df["tweet"])

x_train = pd.DataFrame(train_tweets_countvectorizer.toarray())
x_test = pd.DataFrame(test_tweets_countvectorizer.toarray())

y_train = train_tweets_df["label"]

from sklearn.naive_bayes import MultinomialNB

NB_classifier = MultinomialNB()
NB_classifier.fit(x_train, y_train)
y_predict_test = NB_classifier.predict(x_test)
y_predict_test
sentiment_predicted_labels = pd.DataFrame({"id":test_tweets_df["id"], 'label': y_predict_test})
sentiment_predicted_labels
sentiment_predicted_labels.to_csv("../datasets/analytics_vidhya/sentiment_analysis/submission_twitter.csv", index=False)
