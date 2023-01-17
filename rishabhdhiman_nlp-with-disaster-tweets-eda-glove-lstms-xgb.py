import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import nltk
import missingno as msno
from wordcloud import WordCloud, STOPWORDS
import string
import re
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.initializers import Constant
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

%matplotlib inline
train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
sample = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
print(train.shape, test.shape)
# Plotting the missing values
msno.bar(train)
# Percentage missing values in keyword, location
def pert(data):
  return (len(data[data["keyword"].isnull() == True])/data.shape[0]) * 100

def pert_loc(data):
  return (len(data[data["location"].isnull() == True])/data.shape[0]) * 100

print("percentage of keyword values missing in train --> {}".format(pert(train)))
print("percentage of keyword values missing in test --> {}".format(pert(test)))
print("percentage of location values missing in train --> {}".format(pert_loc(train)))
print("percentage of location values missing in test --> {}".format(pert_loc(test)))
print("\n")
print("Percentage of missing values in both train and test looks quite similar.\
May be both train and test data come from same sample")
# Plottin the most repetitive words in "text" column
stopwords = set(STOPWORDS)
def word_cloud(data, title = None):
  cloud = WordCloud(
      background_color = "black",
      stopwords = stopwords,
      max_words=200,
      max_font_size=40, 
      scale=3,
  ).generate(str(data))
  fig = plt.figure(figsize= (15, 15))
  plt.axis("off")
  if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.25)

  plt.imshow(cloud)
  plt.show()
word_cloud(train["text"], "Most repeated words in train['text']")
word_cloud(test["text"], "Most repeated words in test['text']")
# Target count
plt.figure(figsize = (10, 8))
uniques = train["target"].value_counts()
sns.barplot(x = uniques.index, y = uniques.values, data = uniques)
plt.xlabel("Target Values")
plt.ylabel("Count Values")
sns.despine(left = True, bottom = True)
plt.show()
# Most repeated words in real disaster tweets
word_cloud(train[train["target"] == 1]["text"], "Most repeated words in real disaster tweets in train data")
word_cloud(train[train["target"] == 0]["text"], "Most repeated words in fake disaster tweets in train data")
l1 = list(train["keyword"].unique())
l2 = list(test["keyword"].unique())
if l1 == l2:
  print("Two lists are identical")
else:
  print("Two lists are not identical")
# Distribution of keywords in real and fake tweets 
plt.figure(figsize = (10, 80), dpi = 100)
plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=12)
sns.countplot(y = "keyword", hue = "target", data = train)
plt.legend(loc = 1)
plt.show()
# Number of words in the text 
train["num_words"] = train["text"].apply(lambda x : len(str(x).split()))
test["num_words"] = test["text"].apply(lambda x : len(str(x).split()))

# Number of characters
train["num_chars"] = train["text"].apply(lambda x : len(str(x)))
test["num_chars"] = test["text"].apply(lambda x : len(str(x)))

# Number of stopwords
train["num_stopwords"] = train["text"].apply(lambda x : len([word for word in str(x).lower().split()\
                                                             if word in stopwords]))
test["num_stopwords"] = test["text"].apply(lambda x : len([word for word in str(x).lower().split()\
                                                             if word in stopwords]))

# Number of punctuations
train["num_punctuation"] = train["text"].apply(lambda x : len([p for p in x.split() if p in string.punctuation]))
test["num_punctuation"] = test["text"].apply(lambda x : len([p for p in x.split() if p in string.punctuation]))
fig, axes = plt.subplots(4, 1, figsize = (10, 20))

# num_words
sns.boxplot(x = "target", y = "num_words", data = train, ax = axes[0])
axes[0].set_xlabel('Target', fontsize=12)
axes[0].set_title("Number of words in each class", fontsize=15)

# num_chars
sns.boxplot(x = "target", y = "num_chars", data = train, ax = axes[1])
axes[1].set_xlabel('Target', fontsize=12)
axes[1].set_title("Number of characters in each class", fontsize=15)

# num_stopwords
sns.boxplot(x = "target", y = "num_stopwords", data = train, ax = axes[2])
axes[2].set_xlabel('Target', fontsize=12)
axes[2].set_title("Number of stopwords in each class", fontsize=15)

# num_punctuation
sns.boxplot(x = "target", y = "num_punctuation", data = train, ax = axes[3])
axes[3].set_xlabel('Target', fontsize=12)
axes[3].set_title("Number of punctuations in each class", fontsize=15)
from collections import defaultdict

train0 = train[train["target"] == 0]
train1 = train[train["target"] == 1]

## custom function for ngram generation ##
def generate_ngrams(text, n_gram=1):
    
    token = [token for token in text.lower().split() if token != "" if token not in stopwords]
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    return [" ".join(ngram) for ngram in ngrams]


freq_dict = defaultdict(int)
for sent in train0["text"]:
  for word in generate_ngrams(sent):
    freq_dict[word] += 1
fd_sorted0 = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted0.columns = ["word", "wordcount"]


freq_dict = defaultdict(int)
for sent in train1["text"]:
  for word in generate_ngrams(sent):
    freq_dict[word] += 1
fd_sorted1 = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted1.columns = ["word", "wordcount"]

fig, axes = plt.subplots(1, 2, figsize = (12, 12))
plt.tight_layout()
sns.despine()
for i in range(2):
    sns.barplot(x = "wordcount", y = "word", data = globals()["fd_sorted" + str(i)].iloc[:50, :], ax = axes[i])
    axes[i].set_xlabel('Count', fontsize=12)
    axes[i].set_title(f"Most repetitive words in {i} class", fontsize=15)
train0 = train[train["target"] == 0]
train1 = train[train["target"] == 1]

## custom function for ngram generation ##
def generate_ngrams(text, n_gram=2):
    
    token = [token for token in text.lower().split() if token != "" if token not in stopwords]
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    return [" ".join(ngram) for ngram in ngrams]


freq_dict = defaultdict(int)
for sent in train0["text"]:
  for word in generate_ngrams(sent):
    freq_dict[word] += 1
fd_sorted0 = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted0.columns = ["word", "wordcount"]


freq_dict = defaultdict(int)
for sent in train1["text"]:
  for word in generate_ngrams(sent):
    freq_dict[word] += 1
fd_sorted1 = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted1.columns = ["word", "wordcount"]

fig, axes = plt.subplots(1, 2, figsize = (12, 12))
plt.tight_layout()
sns.despine()
for i in range(2):
    sns.barplot(x = "wordcount", y = "word", data = globals()["fd_sorted" + str(i)].iloc[:50, :], ax = axes[i])
    axes[i].set_xlabel('Count', fontsize=12)
    axes[i].set_title(f"Most repetitive words in {i} class", fontsize=15)
train0 = train[train["target"] == 0]
train1 = train[train["target"] == 1]

## custom function for ngram generation ##
def generate_ngrams(text, n_gram=3):
    
    token = [token for token in text.lower().split() if token != "" if token not in stopwords]
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    return [" ".join(ngram) for ngram in ngrams]


freq_dict = defaultdict(int)
for sent in train0["text"]:
  for word in generate_ngrams(sent):
    freq_dict[word] += 1
fd_sorted0 = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted0.columns = ["word", "wordcount"]


freq_dict = defaultdict(int)
for sent in train1["text"]:
  for word in generate_ngrams(sent):
    freq_dict[word] += 1
fd_sorted1 = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted1.columns = ["word", "wordcount"]

fig, axes = plt.subplots(1, 2, figsize = (12, 12))
plt.tight_layout()
sns.despine()
for i in range(2):
    sns.barplot(x = "wordcount", y = "word", data = globals()["fd_sorted" + str(i)].iloc[:50, :], ax = axes[i])
    axes[i].set_xlabel('Count', fontsize=12)
    axes[i].set_title(f"Most repetitive words in {i} class", fontsize=15)
%%time

glove_embeddings = np.load('../input/pickled-glove840b300d-for-10sec-loading/glove.840B.300d.pkl', allow_pickle=True)
df = train.append(test, ignore_index = True)
def build_vocab(X):
    
    tweets = X.apply(lambda s: s.split()).values      
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
    return covered, oov, n_covered, n_oov

covered, oov, n_covered, n_oov = check_embeddings_coverage(df["text"], glove_embeddings)
print(f"Number of words covered by Glove embeddings --> {n_covered}")
print(f"Number of words not covered by Glove embeddings --> {n_oov}")
print(f"Percentage of words covered by Glove embeddings --> {(n_covered/(n_covered + n_oov)) * 100}%")
df["text"] = df["text"].apply(lambda x : x.lower())
df["keyword"].fillna("keyword", inplace = True)
df["text"] = df["text"] + " " + df["keyword"]
df.drop(["keyword", "location"], axis = 1, inplace = True)
list_all_words = " ".join(df["text"])
not_english = [word for word in list_all_words.split() if word.isalpha() == False]

def clean_data(data):
    # Remove urls
    data = re.sub(r'https?\S+', '', data)
    # Remove html tags
    data = re.sub(r"<.*?>", "", data)
    # Remove punctuations
    t = [w for w in data if w not in string.punctuation]
    data = "".join(t)
    # Remove stopwords
    t = [w for w in data.split() if w not in stopwords]
    data = " ".join(t)
    # Removing numbers from text
    data = re.sub(r"\d+", "", data)

    data = re.sub(r"\x89Û_", "", data)
    data = re.sub(r"\x89ÛÒ", "", data)
    data = re.sub(r"\x89ÛÓ", "", data)
    data = re.sub(r"\x89ÛÏWhen", "When", data)
    data = re.sub(r"\x89ÛÏ", "", data)
    data = re.sub(r"China\x89Ûªs", "China's", data)
    data = re.sub(r"let\x89Ûªs", "let's", data)
    data = re.sub(r"\x89Û÷", "", data)
    data = re.sub(r"\x89Ûª", "", data)
    data = re.sub(r"\x89Û\x9d", "", data)
    data = re.sub(r"å_", "", data)
    data = re.sub(r"\x89Û¢", "", data)
    data = re.sub(r"\x89Û¢åÊ", "", data)
    data = re.sub(r"fromåÊwounds", "from wounds", data)
    data = re.sub(r"åÊ", "", data)
    data = re.sub(r"åÈ", "", data)
    data = re.sub(r"JapÌ_n", "Japan", data)    
    data = re.sub(r"Ì©", "e", data)
    data = re.sub(r"å¨", "", data)
    data = re.sub(r"SuruÌ¤", "Suruc", data)
    data = re.sub(r"åÇ", "", data)
    data = re.sub(r"å£3million", "3 million", data)
    data = re.sub(r"åÀ", "", data)
    
    # Remove words not alphabets
    t = [w for w in data.split() if w not in not_english]
    data = " ".join(t)
    
    return data    


df["text"] = df["text"].apply(lambda x : clean_data(x))
covered, oov, n_covered, n_oov = check_embeddings_coverage(df["text"], glove_embeddings)
print(f"Number of words covered by Glove embeddings --> {n_covered}")
print(f"Number of words not covered by Glove embeddings --> {n_oov}")
print(f"Percentage of words covered by Glove embeddings --> {(n_covered/(n_covered + n_oov)) * 100}%")
embed_size = 300 # how big is each word vector
maxlen = 20 # max number of words in a comment to use
max_features = 20000
tokenizer = Tokenizer(oov_token = "<OOV>", num_words = max_features)
tokenizer.fit_on_texts(df["text"])
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(df["text"])
padded = pad_sequences(sequences, padding = "post", maxlen = maxlen)
train_x = padded[:7613, :]
test = padded[7613:, :]
train_y = df[df["target"].isnull() == False]["target"].apply(int).values.reshape(-1, 1)
num_words = min(max_features, len(word_index)) + 1
embedding_dim = 300
# first create a matrix of zeros, this is our embedding matrix
embedding_matrix = np.zeros((num_words, embedding_dim))
# for each word in out tokenizer lets try to find that work in our w2v model
for word, i in word_index.items():
    if i > max_features:
        continue
    embedding_vector = covered.get(word)
    if embedding_vector is not None:
        # we found the word - add that words vector to the matrix
        embedding_matrix[i] = embedding_vector
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(num_words,
                    embedding_dim,
                    embeddings_initializer=Constant(embedding_matrix),
                    input_length=maxlen,
                    trainable=False),
    tf.keras.layers.SpatialDropout1D(0.2),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dropout(0.10),
    tf.keras.layers.Dense(units=32, activation="relu"),
    tf.keras.layers.Dense(units=8, activation="relu"),
    tf.keras.layers.Dense(units=1, activation="sigmoid")
    
    
])

model.compile(loss = "binary_crossentropy", optimizer='adam', metrics = ["accuracy"])
model.summary()
batch_size = 128
num_epochs = 20

history = model.fit(train_x, train_y, batch_size = batch_size, epochs = num_epochs)
# Make predictions
# Preparing test data
y_pred = model.predict(test)

model_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
model_submission['target'] = np.round(y_pred).astype('int')
model_submission.to_csv('model_submission2.csv', index=False)
pad_df = pd.DataFrame(padded)
new_df = pd.concat([df, pad_df], axis=1)
new_df.drop("text", inplace = True, axis = 1)
train_new_df = new_df[new_df["target"].isnull() == False]
test_new_df = new_df[new_df["target"].isnull() == True]
test_new_df.drop("target", inplace = True, axis = 1)
X = train_new_df.drop("target", axis = 1).values
y = train_new_df["target"].apply(int).values.reshape(-1, 1)
# It will take a while to run. I have already run this on my local host. So I am just writing the code here.

param_test = {
    "max_depth":range(3,10,2),
    "min_child_weight":range(1,6,2),
    "gamma":[i/10.0 for i in range(0,5)]
}


gsearch = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
                                             min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                             objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
                                             param_grid = param_test, n_jobs=4,iid=False, cv=5)

gsearch.fit(X,y)
gsearch.best_params_
xgb = XGBClassifier(
     learning_rate =0.1,
     n_estimators=140,
     max_depth=5,
     min_child_weight=1,
     gamma=0,
     subsample=0.8,
     colsample_bytree=0.8,
     objective= 'binary:logistic',
     nthread=4,
     scale_pos_weight=1,
     seed=27)
xgb.fit(X, y)
preds = xgb.predict(test_new_df.values)
xgb_pred = model.predict(test)

xgb_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
xgb_submission['target'] = np.round(y_pred).astype('int')
xgb_submission.to_csv('xgb_submission.csv', index=False)