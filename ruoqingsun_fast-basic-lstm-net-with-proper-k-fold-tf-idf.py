import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

print("Loading data...")
train = pd.read_csv("../input/movie-review-sentiment-analysis-kernels-only/train.tsv", sep="\t")
print("Train shape:", train.shape)
test = pd.read_csv("../input/movie-review-sentiment-analysis-kernels-only/test.tsv", sep="\t")
print("Test shape:", test.shape)

enc = OneHotEncoder(sparse=False)
enc.fit(train["Sentiment"].values.reshape(-1, 1))
print("Number of classes:", enc.n_values_[0])

print("Class distribution:\n{}".format(train["Sentiment"].value_counts()/train.shape[0]))
print("Ratio of test set examples which occur in the train set: {0:.2f}".format(len(set(train["Phrase"]).intersection(set(test["Phrase"])))/test.shape[0]))
test = pd.merge(test, train[["Phrase", "Sentiment"]], on="Phrase", how="left")
print("Number of unique sentence in train: ", train.SentenceId.nunique())
print("Number of unique sentence in test: ", test.SentenceId.nunique())

sentence_distribution = train.groupby(['SentenceId'])['Sentiment'].agg(['min','max','count','nunique']).reset_index().sort_values("nunique", ascending = False)
sentence_distribution[sentence_distribution['nunique'] >= 4]['SentenceId'].nunique()
train[train['SentenceId']==4268]
from sklearn.feature_extraction.text import CountVectorizer

cv1 = CountVectorizer()
cv1.fit(train["Phrase"])

cv2 = CountVectorizer()
cv2.fit(test["Phrase"])

print("Train Set Vocabulary Size:", len(cv1.vocabulary_))
print("Test Set Vocabulary Size:", len(cv2.vocabulary_))
print("Number of Words that occur in both:", len(set(cv1.vocabulary_.keys()).intersection(set(cv2.vocabulary_.keys()))))
import math
def computeTF():
    dicts = []
    for i in range(0, 5):
        tf_dict = {}
        sentences = list(train[train['Sentiment']==i]['Phrase'].values)
        for sentence in sentences:
            sentence = sentence.lower()
            words = sentence.split()
            for word in words:
                if word not in tf_dict:
                    tf_dict[word] = 1
                else:
                    tf_dict[word] += 1
        total_words = sum(tf_dict.values())
        for word, val in tf_dict.items():
            tf_dict[word] = val * 1.0/total_words
        dicts.append(tf_dict)
    return dicts
def computeIDF(tf_dicts):
    keys_all = []
    idf_dict = {}
    for i in range(0, 5):
        keys_all += list(tf_dicts[i].keys())
    for key in keys_all:
        if key not in idf_dict:
            idf_dict[key] = 1
        else:
            idf_dict[key] += 1
    for word, val in idf_dict.items():
            idf_dict[word] = math.log(5.0 / idf_dict[word])
    return idf_dict
def computeTFIDF(tf_dicts, idf_dict):
    tfidf_dicts = []
    for i in range(0, 5):
        tfidf_dict = {}
        for word, val in tf_dicts[i].items():
            tfidf_dict[word] = tf_dicts[i][word] * idf_dict[word]
        tfidf_dicts.append(tfidf_dict)
    return tfidf_dicts
tf_dicts = computeTF()
idf_dict = computeIDF(tf_dicts)
tfidf_dicts = computeTFIDF(tf_dicts, idf_dict)

keywords = []
for i in range(0, 5):
    important_words = sorted(tfidf_dicts[i].items(), key=lambda x: x[1], reverse=True)[1:100]
    keywords.append([item[0] for item in important_words])
all_keywords = [keyword for keyword_list in keywords for keyword in keyword_list ]
def transform(df):
    df["phrase_count"] = df.groupby("SentenceId")["Phrase"].transform("count")
    df["word_count"] = df["Phrase"].apply(lambda x: len(x.split()))
    df["has_upper"] = df["Phrase"].apply(lambda x: x.lower() != x)
    df["sentence_end"] = df["Phrase"].apply(lambda x: x.endswith("."))
    df["after_comma"] = df["Phrase"].apply(lambda x: x.startswith(","))
    df["sentence_start"] = df["Phrase"].apply(lambda x: "A" <= x[0] <= "Z")
    df["Phrase"] = df["Phrase"].apply(lambda x: x.lower())
    df["sentiment0_words"] = df["Phrase"].apply(lambda x: len(set(x.split()).intersection(set(keywords[0]))))
    df["sentiment1_words"] = df["Phrase"].apply(lambda x: len(set(x.split()).intersection(set(keywords[1]))))
    df["sentiment3_words"] = df["Phrase"].apply(lambda x: len(set(x.split()).intersection(set(keywords[3]))))
    df["sentiment4_words"] = df["Phrase"].apply(lambda x: len(set(x.split()).intersection(set(keywords[4]))))
    df["no_sentiment_words"] = df["Phrase"].apply(lambda x: len(set(x.split()))-len(set(x.split()).intersection(set(all_keywords))))
    return df

train = transform(train)
test = transform(test)

dense_features = ["phrase_count", "word_count", "has_upper", "after_comma", "sentence_start", "sentence_end", 
                  "sentiment0_words", "sentiment1_words","sentiment3_words", "sentiment4_words", "no_sentiment_words"]

train.groupby("Sentiment")[dense_features].mean().reset_index()
print(max(train["phrase_count"]))
print(max(train["word_count"]))
NUM_FOLDS = 5

train["fold_id"] = train["PhraseId"].apply(lambda x: x%NUM_FOLDS)
EMBEDDING_FILE = "../input/glove-global-vectors-for-word-representation/glove.6B.100d.txt"
EMBEDDING_DIM = 100

all_words = set(cv1.vocabulary_.keys()).union(set(cv2.vocabulary_.keys()))

def get_embedding():
    embeddings_index = {}
    f = open(EMBEDDING_FILE)
    for line in f:
        values = line.split()
        word = values[0]
        if len(values) == EMBEDDING_DIM + 1 and word in all_words:
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs
    f.close()
    return embeddings_index

embeddings_index = get_embedding()
print("Number of words in total:", len(all_words))
print("Number of words that don't exist in GLOVE:", len(all_words - set(embeddings_index)), ", {0:.2f}".format(len(all_words - set(embeddings_index))/len(all_words)), "of all words")
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

MAX_SEQUENCE_LENGTH = 60

tokenizer = Tokenizer(filters="")
tokenizer.fit_on_texts(np.append(train["Phrase"].values, test["Phrase"].values))
word_index = tokenizer.word_index

nb_words = len(word_index) + 1
embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))


for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
        
seq = pad_sequences(tokenizer.texts_to_sequences(train["Phrase"]), maxlen=MAX_SEQUENCE_LENGTH)
test_seq = pad_sequences(tokenizer.texts_to_sequences(test["Phrase"]), maxlen=MAX_SEQUENCE_LENGTH)
from keras.layers import *
from keras.models import Model
from keras.callbacks import EarlyStopping
from imblearn.over_sampling import SMOTE, ADASYN

def build_model():
    embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)
    dropout = SpatialDropout1D(0.2)
    mask_layer = Masking()
    lstm_layer = LSTM(50)
    
    seq_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype="int32")
    dense_input = Input(shape=(len(dense_features),))
    
    dense_vector = BatchNormalization()(dense_input)
    
    phrase_vector = lstm_layer(mask_layer(dropout(embedding_layer(seq_input))))
    
    feature_vector = concatenate([phrase_vector, dense_vector])
    feature_vector = Dense(50, activation="relu")(feature_vector)
    feature_vector = Dense(20, activation="relu")(feature_vector)
    
    output = Dense(5, activation="softmax")(feature_vector)
    
    model = Model(inputs=[seq_input, dense_input], outputs=output)
    return model
# Resampling

# print(train_seq.shape)
# print(train_dense.shape)
# new_dataframe = pd.concat([
#     train_dense,
#     pd.DataFrame(train_seq, dtype=np.float64)
# ], axis=1, ignore_index=True)
# new_dataframe.head()

# from imblearn.over_sampling import SMOTE, ADASYN
# X_resampled, y_resampled = SMOTE().fit_sample(train_seq, train[train["fold_id"] != 0]["Sentiment"].values)
# from collections import Counter
# print(sorted(Counter(y_resampled).items()))
test_preds = np.zeros((test.shape[0], 5))

for i in range(NUM_FOLDS):
    print("FOLD", i+1)
    
    print("Splitting the data into train and validation...")
    train_seq, val_seq = seq[train["fold_id"] != i], seq[train["fold_id"] == i]
    train_dense, val_dense = train[train["fold_id"] != i][dense_features], train[train["fold_id"] == i][dense_features]
    
    y_train = train[train["fold_id"] != i]["Sentiment"].values
    y_val = train[train["fold_id"] == i]["Sentiment"].values
    y_train = enc.transform(train[train["fold_id"] != i]["Sentiment"].values.reshape(-1, 1))
    y_val = enc.transform(train[train["fold_id"] == i]["Sentiment"].values.reshape(-1, 1))
    
    print("Building the model...")
    model = build_model()
    model.compile(loss="categorical_crossentropy", optimizer="nadam", metrics=["acc"])
    
    ## Early stopping
    early_stopping = EarlyStopping(monitor="val_acc", patience=2, verbose=2) # verbose = 1 ==> print out every line, which takes time
    
    print("Training the model...")
    model.fit([train_seq, train_dense], y_train, validation_data=([val_seq, val_dense], y_val),
              epochs=15, batch_size=1024, shuffle=True, callbacks=[early_stopping], verbose=2)
    
    print("Predicting...")
    test_preds += model.predict([test_seq, test[dense_features]], batch_size=1024, verbose=2)
    print()
    
test_preds /= NUM_FOLDS
model.summary()
print("Select the class with the highest probability as prediction...")
test["pred"] = test_preds.argmax(axis=1)

print("Use these predictions for the phrases which don't exist in train set...")
test.loc[test["Sentiment"].isnull(), "Sentiment"] = test.loc[test["Sentiment"].isnull(), "pred"]

print("Make the submission ready...")
test["Sentiment"] = test["Sentiment"].astype(int)
test[["PhraseId", "Sentiment"]].to_csv("submission.csv", index=False)
