# basics
from collections import defaultdict
import re

# Numerics, data processing & similar utils
import numpy as np
import pandas as pd

# nlp tools
import nltk
from nltk.corpus import stopwords
import spacy

# Machine learning (metrics from scikit-learn, model building tools from Keras on tensorflow backend)
from keras import backend as K
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.layers import Conv1D, CuDNNGRU, CuDNNLSTM, GlobalAveragePooling1D, GlobalMaxPooling1D, Dense, Embedding, Flatten
from keras.layers import Activation, BatchNormalization, Bidirectional, concatenate, Dropout, Input, SpatialDropout1D
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import Sequence, pad_sequences
from keras.regularizers import l1_l2
from keras.utils import plot_model
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# Visualisation
import matplotlib.pyplot as plt
%matplotlib inline
import pydot
from tqdm import tqdm, tqdm_notebook
nltk.download("stopwords")
tqdm_notebook().pandas()
TRAIN_FILE = "../data/train.csv"
TEST_DATA_FILE = "../data/test.csv"
EMB_FASTTEXT_300_FILE = "../data/wiki.en.vec"

train = pd.read_csv(TRAIN_FILE)
test_data = pd.read_csv(TEST_DATA_FILE)
print("Training set dimension: " + str(train.shape))
print("Test set dimension: " + str(test_data.shape))
train.head(5)
np.sum(train.isna())
train_size = train.shape[0]
test_size = test_data.shape[0]
print(f"Number of examples in the training set: {train_size}")
print(f"Number of examples in the test set: {test_size}")
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
percent_toxic = train[LABELS].sum().apply(lambda x: (x / train_size) * 100)
print("Toxic comments by category (as fractions of the whole dataset): \n")
for category, percent in percent_toxic.items():
    print(category + ": " + str(round(percent, 2)) + "%")
ts = train[LABELS].sum()
print("Total toxicity occurrences detected by human annotators: " + str(ts.sum()))
train_target = train[LABELS]
train_comments = train["comment_text"]
test_comments = test_data["comment_text"]
CLEANING_DICT = {
    " ": ["\n", "/", "(([0-9]{2,3}[^a-z0-9])+)+", "\[\[.*\]", " +"],
    " as far as i know ": ["afaik"],
    " in my opinion ": ["imo", "imho"],
    " sad ": ["\:\(", "\:\'\(", "\:\(+"],
    " lol ": [" (h[ae])+h? ", " lel ", " kek ", "lu+lz?", "loo(o)*lz?"],
    " happy ": ["\:\)+", ";\)+", "\:d+"],
    " american ": ["amerikan"],
    " adolf ": ["adolf"],
    " hitler ": ["hitler"],
    "fuck": ["(f)(u|[^a-z0-9 ])(c|[^a-z0-9 ])(k|[^a-z0-9 ])", "(f)([^a-z]*)(u)([^a-z]*)(c)([^a-z]*)(k)",
            " f[!@#\$%\^\&\*]*u[!@#\$%\^&\*]*k", "f u u c", "(f)(c|[^a-z ])(u|[^a-z ])(k)", r"f\*", "feck ",
            " fux ", "f\*\*", "f\.u\.", "f###", " fu ", "f@ck", "f u c k", "f uck", "f ck"],
    "fucking ": ["f[u|a]c?king?"],
    " ass ": ["[^a-z]ass ", "[^a-z]azz ", "arrse", " arse ", "@\$\$", "[^a-z]anus", " a\*s\*s", "[^a-z]ass[^a-z ]",
             "a[@#\$%\^&\*][@#\$%\^&\*]", "[^a-z]anal ", "a s s", "butt "],
    " asshole ": [" a[s|z]*wipe", "a[s|z]*[w]*h[o|0]+[l]*e", "@\$\$hole"],
    " bitch ": ["b[w]*i[t]*ch", "b!tch", "bi\+ch", "b!\+ch", "(b)([^a-z]*)(i)([^a-z]*)(t)([^a-z]*)(c)([^a-z]*)(h)",
                "biatch", "bi\*\*h", "bytch", "b i t c h"],
    " bastard ": ["ba[s|z]t[a|e]rd"],
    " gay ": ["gay"],
    " cock ": ["[^a-z]cock", "c0ck", "[^a-z]cok ", "c0k", "[^a-z]cok[^aeiou]", " cawk",
               "(c)([^a-z ])(o)([^a-z ]*)(c)([^a-z ]*)(k)", "c o c k"],
    " dick ": [" dick[aeiou]", "deek", "d i c k"],
    " suck ": ["(s)([^a-z ]*)(u)([^a-z ]*)(c)([^a-z ]*)(k)", "sucks", "5uck", "s u c k"],
    " sucking ": ["s[u|a]c?king?"],
    " cunt ": ["cunt", "c u n t", "c\*\*\*", "c\*\*t", "c\*nt"],
    " bullshit ": ["bullsh\*t", "bull\$hit"],
    " homosexual ": ["homo"],
    " jerk ": ["jerk"],
    " idiot ": ["i[d]+io[t]+", "(i)([^a-z ]*)(d)([^a-z ]*)(i)([^a-z ]*)(o)([^a-z ]*)(t)", "idiots", "i d i o t"],
    " dumb ": ["(d)([^a-z ]*)(u)([^a-z ]*)(m)([^a-z ]*)(b)", "d\*mb", "$dumm"],
    " shit ": ["shitty", "(s)([^a-z ]*)(h)([^a-z ]*)(i)([^a-z ]*)(t)", "shite", "\$hit", "s h i t"],
    " shit hole ": ["shythole", "shithole"],
    " retard ": ["returd", "retad", "retard", "ret[au]rded", "wiktard", "wikitard", "wikitud"],
    " dumb ass": ["dumbass", "dubass", "du(m)+ass"],
    " ass head ": ["butthead"],
    " sex ": ["s3x", "s\*x"],
    " nigger ": ["nigger", "ni[g]+a", " nigr ", "negrito", "niguh", "n3gr", "n i g g e r"],
    " shut the fuck up ": ["stfu"],
    " rape ": ["reap", "rpe"],
    " pussy ": ["pussy[^c]", "pusy", "pussi[^l]", "pusses"],
    " faggot ": ["faggot", " fa[g]+[s]*[^a-z ]", "fagot", "f a g g o t", "faggit",
                 "(f)([^a-z ]*)(a)([^a-z ]*)([g]+)([^a-z ]*)(o)([^a-z ]*)(t)", "fau[g]+ot", "fae[g]+ot",
                 "fagg+az", "fagg+otz"],
    " motherfucker ": [" motha ", "motha f ", "moth[a|er]fucka?r", "mother f", "motherucker"],
    " whore ": ["w\*\*\*(\*)?", "whor", "w h o r e"],
    " and ": ["&"],
    " at ": [" @ "],
    "ing ": ["$in\'"]
}
STOPWORDS = stopwords.words("english")
SPACY_LANG_MODEL = spacy.load("en_core_web_md")
def clean_text(comment, replacement_dict=CLEANING_DICT, spacy_model=SPACY_LANG_MODEL, stopwds=STOPWORDS):
    comment = comment.lower()
    for base, patterns in replacement_dict.items():
        for pattern in patterns:
            comment = re.sub(pattern, base, comment)
    tokenized = list(spacy_model(comment, disable=["tagger", "parser", "ner"]))
    lemmatized = [token.lemma_ if token.lemma_ != "-PRON-" else token.lower_ for token in tokenized]
    lemmatized = [lem for lem in lemmatized if lem not in stopwds and lem != " "]
    comment = " ".join(lemmatized)
    return comment
train_clean = train_comments.progress_apply(clean_text)
train_clean.to_csv("../data/train_clean.csv", index=False, header=["comment_text"])
test_clean = test_comments.progress_apply(clean_text)
test_clean.to_csv("../data/test_clean.csv", index=False, header=["comment_text"])
train_clean = pd.read_csv("../data/train_clean.csv", header=0)
test_clean = pd.read_csv("../data/test_clean.csv", header=0)
train_clean = train_clean["comment_text"]
test_clean = test_clean["comment_text"]
MAX_FEATURES = 300000
MAX_COMMENT_LEN = 900
def tokenize_for_keras(tr_set, test_set, max_features):
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(tr_set) + list(test_set))
    train_tokenized = tokenizer.texts_to_sequences(tr_set)
    test_tokenized = tokenizer.texts_to_sequences(test_set)
    tokenizer_index = tokenizer.word_index
    return (tokenizer_index, train_tokenized, test_tokenized)
train_feed = train_clean.fillna("fillna")
test_feed = test_clean.fillna("fillna")
word_index, train_tokenized, test_tokenized = tokenize_for_keras(train_feed, test_feed, max_features=MAX_FEATURES)
def get_embedding_dict(emb_file=EMB_FASTTEXT_300_FILE):
    emb_dict = defaultdict()
    with open(emb_file) as file:
        for line in file:
            emb = line.rstrip().rsplit(" ")
            word = emb[0]
            vec = np.asarray(emb[1:], dtype="float32")
            emb_dict[word] = vec
    return emb_dict
emb_dict_fasttext = get_embedding_dict()
EMB_SIZE_FASTTEXT = 300
train_tokenized = pad_sequences(train_tokenized, maxlen=MAX_COMMENT_LEN)
test_tokenized = pad_sequences(test_tokenized, maxlen=MAX_COMMENT_LEN)
def get_embedding_matrix(emb_dict, word_index, max_features, emb_size, gaussian_initialization=False):
    n_words = min(MAX_FEATURES, len(word_index))
    
    if gaussian_initialization:
        stacked_embs = np.stack(emb_dict.values())
        emb_mean, emb_std = (np.mean(stacked_embs), np.std(stacked_embs))
        emb_matrix = np.random.normal(emb_mean, emb_std, size=(n_words, emb_size))
    else:
        emb_matrix = np.zeros((n_words, emb_size))
    
    for word, index in word_index.items():
        if index >= max_features:
            continue
        vec = emb_dict.get(word)
        if vec is not None:
            emb_matrix[index] = vec
    return emb_matrix
emb_matrix_fasttext = get_embedding_matrix(emb_dict=emb_dict_fasttext, word_index=word_index, max_features=MAX_FEATURES, emb_size=EMB_SIZE_FASTTEXT, gaussian_initialization=False)
class ROC_AUC_Score(Callback):
    """Custom Keras callback class tracking ROC AUC.
    After every epoch's end, it prints out current ROC AUC score for model's predictions
    and saves the value to model's logs for potential use by other callbacks.
    """
    
    def __init__(self, validation_data=(), interval=1):
        super().__init__()
        self.interval = interval
        self.val_X, self.val_target = validation_data
     
    def on_train_begin(self, logs={}):
        self.aucs = []
        self.losses = []
    
    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            preds = self.model.predict(self.val_X, verbose=0)
            auc = roc_auc_score(self.val_target, preds)
            self.aucs.append(auc)
            logs["roc_auc_val"] = auc
            print(f"Epoch: {epoch+1} - ROC AUC score: {round(auc, 5)}")

def build_model():
    model_input = Input(shape=(MAX_COMMENT_LEN,))
    X = Embedding(input_dim=MAX_FEATURES, output_dim=EMB_SIZE_FASTTEXT, weights=[emb_matrix_fasttext], trainable=False)(model_input)
    X = SpatialDropout1D(rate=0.5)(X)
    
    X = Dense(units=128, activation="relu")(X)
    X = BatchNormalization()(X)
    X = Dropout(rate=0.5)(X)
    
    Y = Bidirectional(CuDNNLSTM(units=64, return_sequences=True))(X)
    Y = Conv1D(filters=128, kernel_size=3, strides=1, padding="valid")(Y)
    maxpool2 = GlobalMaxPooling1D()(Y)
    avgpool2 = GlobalAveragePooling1D()(Y)
    
    X = Bidirectional(CuDNNGRU(units=128, return_sequences=True))(X)
    X = Conv1D(filters=64, kernel_size=3, strides=1, padding="valid")(X)
    maxpool = GlobalMaxPooling1D()(X)
    avgpool = GlobalAveragePooling1D()(X)
    
    X = concatenate([avgpool, maxpool, avgpool2, maxpool2])
    
    X = Dense(units=128, activation="relu")(X)
    X = BatchNormalization()(X)
    X = Dropout(rate=0.5)(X)
    X = Dense(units=6, activation="sigmoid")(X)

    model = Model(inputs=model_input, outputs=X)
    opt = Adam(lr=0.001, clipvalue=1.0)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
    return model
model = build_model()
X_tr, X_val, y_train, y_val = train_test_split(train_tokenized, train_target, train_size=0.9, random_state=233)
# ROC AUC tracker, model checkpointer
roc_auc = ROC_AUC_Score(validation_data=(X_val, y_val))
model_path="final-model"
checkpointer = ModelCheckpoint(model_path, monitor="roc_auc_val", mode="max", verbose=1, save_best_only=True)
model.fit(x=X_tr, y=y_train, batch_size=64, epochs=15, validation_data=(X_val, y_val), callbacks=[roc_auc, checkpointer])
model = load_model("final-model.h5")
pred_test = model.predict(x=[test_tokenized], batch_size=1024, verbose=1)
def predictions_to_csv(predictions, submission_name):
    submission = pd.read_csv("../data/sample_submission.csv")
    submission[LABELS] = predictions
    submission.to_csv("../output/" + submission_name + ".csv", index=False)
predictions_to_csv(pred_test, "final-submission")
def backup_model(model, model_name):
    model_path = "../models/" + model_name + ".h5"
    model.save(model_path)
    picture_path = "../models/" + model_name + ".png"
    plot_model(model, to_file=picture_path)
backup_model(model, "final-model")
