import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from nltk.corpus import stopwords

import re

import string

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer

from scipy.sparse import coo_matrix, hstack

from sklearn.decomposition import TruncatedSVD

from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

import textblob

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV



import warnings

warnings.filterwarnings('ignore')

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import *

from keras.models import Model

from keras.callbacks import EarlyStopping

from keras.utils.vis_utils import plot_model

from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot



import os

from sklearn.metrics import accuracy_score

from sklearn.metrics import roc_auc_score

from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot

from sklearn.metrics import precision_recall_fscore_support

from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, log_loss



import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv("../input/jigsawtraintest/train_jigsaw.csv")

test= pd.read_csv("../input/jigsawtraintest/test_jigsaw.csv")

pos_df = pd.read_csv("../input/pos-neg-files/positive words.txt", sep="\n", header=None)

neg_df = pd.read_csv("../input/pos-neg-files/Negative words.txt", sep="\n", header=None, encoding = "ISO-8859-1")

pos_df.columns = ['words']

neg_df.columns = ['words']
train.shape, test.shape
train.head()
test.head(3)
#check for null values

for df in [train, test]:

    print(df.isna().sum())
labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
for col in labels:

    print(train[col].value_counts())

    print("\n")
stop_words = set(stopwords.words('english'))

for df in [train, test]:

    df['words_length'] = df['comment_text'].apply(lambda x: len(x))    

    #df['pos_words'] = df['comment_text'].apply(lambda x: len([w for w in re.sub("[^\w]", " ",  x).split() if w in pos_df.words.values]))

    #df['neg_words'] = df['comment_text'].apply(lambda x: len([w for w in re.sub("[^\w]", " ",  x).split()  if w in neg_df.words.values]))
train['words_length'].describe(), test['words_length'].describe()
#words length distribution

for col in labels:

    print(train.groupby([col])['words_length'].mean())

    print("\n")
contractions = {

"ain't": "am not ",

"aren't": "are not",

"can't": "cannot",

"can't've": "cannot have",

"'cause": "because",

"could've": "could have",

"couldn't": "could not",

"couldn't've": "could not have",

"didn't": "did not",

"doesn't": "does not",

"don't": "do not",

"hadn't": "had not",

"hadn't've": "had not have",

"hasn't": "has not",

"haven't": "have not",

"he'd": "he would",

"he'd've": "he would have",

"he'll": "he will",

"he'll've": "he shall have",

"he's": "he is",

"how'd": "how did",

"how'd'y": "how do you",

"how'll": "how will",

"how's": "how is",

"i'd": "I would",

"i'd've": "I would have",

"i'll": "I will",

"i'll've": "I shall have",

"i'm": "I am",

"i've": "I have",

"isn't": "is not",

"it'd": "it would",

"it'd've": "it would have",

"it'll": "it will",

"it'll've": "it will have",

"it's": "it is",

"let's": "let us",

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

"should've": "should have",

"shouldn't": "should not",

"shouldn't've": "should not have",

"so've": "so have",

"so's": "so is",

"that'd": "that would",

"that'd've": "that would have",

"that's": "that is",

"there'd": "there would",

"there'd've": "there would have",

"there's": "there is",

"they'd": "they would",

"they'd've": "they would have",

"they'll": "they will",

"they'll've": "they will have",

"they're": "they are",

"they've": "they have",

"to've": "to have",

"wasn't": "was not",

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

"what've": "what have",

"when's": "when is",

"when've": "when have",

"where'd": "where did",

"where's": "where is",

"where've": "where have",

"who'll": "who will",

"who'll've": "who will have",

"who's": "who is",

"who've": "who have",

"why's": "why is",

"why've": "why have",

"will've": "will have",

"won't": "will not",

"won't've": "will not have",

"would've": "would have",

"wouldn't": "would not",

"wouldn't've": "would not have",

"y'all": "you all",

"y'all'd": "you all would",

"y'all'd've": "you all would have",

"y'all're": "you all are",

"y'all've": "you all have",

"you'd": "you would",

"you'd've": "you would have",

"you'll": "you will",

"you'll've": "you will have",

"you're": "you are",

"you've": "you have"

}
def preprocess_text_for_word_embeddings(text):    

    for word in text.split():

        if word.lower() in contractions:

            text = text.replace(word, contractions[word.lower()])    

    text = re.sub(r'\d+', '', text)

    text = re.sub('[^A-Za-z0-9]+', ' ', text)

    text = text.strip()

    

    return text
full_text = [i for i in train['comment_text']] + [i for i in test['comment_text']] 

full_text = [(lambda x: preprocess_text_for_word_embeddings(x))(x) for x in full_text]
train['clean_text'] = full_text[:len(train)]

test['clean_text'] = full_text[len(train):]
train.head()
train['clean_text'].fillna("##", inplace = True)

test['clean_text'].fillna("##", inplace = True)
lower_case_text = [i.lower() for i in train['clean_text']] + [i.lower() for i in test['clean_text']] 
def tfidf_transform(clean_text):

    vectorizer = TfidfVectorizer(stop_words = set(stopwords.words('english')))

    vectorizer.fit(clean_text)

    x_train_tfidf_vec = vectorizer.transform(clean_text[:len(train)])

    x_test_tfidf_vec = vectorizer.transform(clean_text[len(train):])

    print(x_train_tfidf_vec.shape, x_test_tfidf_vec.shape)

    return x_train_tfidf_vec, x_test_tfidf_vec  
x_train_tfidf_vec, x_test_tfidf_vec =  tfidf_transform(lower_case_text)
clf = LogisticRegression()

grid_values = {'penalty': ['l1', 'l2'],'C':[0.001,.009,0.01,.09,1,5,10,25]}

grid_clf = GridSearchCV(clf, param_grid = grid_values,scoring = 'recall')

grid_clf.fit(x_train_tfidf_vec, train['toxic'])
y_pred_acc = grid_clf.predict_proba(x_test_tfidf_vec)[:,1]

roc_auc_score(test['toxic'], y_pred_acc)
y_prediction = pd.DataFrame(columns=labels)

for col in labels:

    model = LogisticRegression()

    model.fit(x_train_tfidf_vec, train[col])

    y_prediction[col] = model.predict_proba(x_test_tfidf_vec)[:,1]

    print(col, " ROC AUC Score = ",roc_auc_score(test[col], y_prediction[col]))
def make_roc_plot(y_true, y_pred):

    [fpr, tpr, thr] = roc_curve(y_true, y_pred)



    idx = np.min(np.where(tpr > 0.95)) # index of the first threshold for which the sensibility > 0.95



    plt.figure()

    plt.plot(fpr, tpr, color='coral', label='ROC curve (area = %0.3f)' % auc(fpr, tpr))

    plt.plot([0, 1], [0, 1], 'k--')

    plt.plot([0,fpr[idx]], [tpr[idx],tpr[idx]], 'k--', color='blue')

    plt.plot([fpr[idx],fpr[idx]], [0,tpr[idx]], 'k--', color='blue')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate (1 - specificity)', fontsize=14)

    plt.ylabel('True Positive Rate (recall)', fontsize=14)

    plt.title('Receiver operating characteristic (ROC) curve')

    plt.legend(loc="lower right")

    plt.show()



    print("Using a threshold of %.3f " % thr[idx] + "guarantees a sensitivity of %.3f " % tpr[idx] +  

      "and a specificity of %.3f" % (1-fpr[idx]) + 

      ", i.e. a false positive rate of %.2f%%." % (np.array(fpr[idx])*100))
make_roc_plot(test['toxic'], y_prediction['toxic'])
make_roc_plot(test['obscene'], y_prediction['obscene'])
svd = TruncatedSVD(n_components = 100)

x_train_tfidf_vec = svd.fit_transform(x_train_tfidf_vec)

x_test_tfidf_vec = svd.fit_transform(x_test_tfidf_vec)
#x_train_tfidf_vec = hstack([coo_matrix(x_train_tfidf_vec), coo_matrix(train[cov_labels])])

#x_test_tfidf_vec = hstack([coo_matrix(x_test_tfidf_vec), coo_matrix(test[cov_labels])])
x_train_tfidf_vec.shape, x_test_tfidf_vec.shape
x_train_tfidf_vec.max()
x_train_vec, x_val_vec, y_train, y_val = train_test_split(x_train_tfidf_vec, train[labels],

                                                    test_size=0.2,

                                                    random_state=0)
x_train_text, x_val_text, _, _ = train_test_split(train['clean_text'], train[labels],

                                                    test_size=0.2,

                                                    random_state=0)
cv1 = CountVectorizer()

cv1.fit(x_train_text)



cv2 = CountVectorizer()

cv2.fit(x_val_text)



cv3 = CountVectorizer()

cv3.fit(test['clean_text'])



print("Train Set Vocabulary Size:", len(cv1.vocabulary_))

print("Val Set Vocabulary Size:", len(cv2.vocabulary_))

print("Test Set Vocabulary Size:", len(cv3.vocabulary_))



print("Number of Words that occur in both:", 

      len(set(cv1.vocabulary_.keys()).intersection(set(cv3.vocabulary_.keys()))))
EMBEDDING_FILE = '../input/glove840b300dtxt/glove.840B.300d.txt'

EMBEDDING_DIM = 300

all_words = set(cv1.vocabulary_.keys()).union(set(cv2.vocabulary_.keys())).union(set(cv3.vocabulary_.keys()))

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

print("Number of words that don't exist in GLOVE:", len(all_words - set(embeddings_index)))
MAX_SEQUENCE_LENGTH = 200
tokenizer = Tokenizer()



np_text =  np.append(x_train_text.values,x_val_text.values)



tokenizer.fit_on_texts(np.append(np_text, test['clean_text'].values))



word_index = tokenizer.word_index



nb_words = len(word_index) + 1

embedding_matrix = np.random.rand(nb_words, EMBEDDING_DIM + 2)



for word, i in word_index.items():

    embedding_vector = embeddings_index.get(word)

    sent = textblob.TextBlob(word).sentiment

    if embedding_vector is not None:

        embedding_matrix[i] = np.append(embedding_vector, [sent.polarity, sent.subjectivity])

    else:

        embedding_matrix[i, -2:] = [sent.polarity, sent.subjectivity]

        

train_seq = pad_sequences(tokenizer.texts_to_sequences(x_train_text), maxlen=MAX_SEQUENCE_LENGTH)

val_seq = pad_sequences(tokenizer.texts_to_sequences(x_val_text), maxlen=MAX_SEQUENCE_LENGTH)

test_seq = pad_sequences(tokenizer.texts_to_sequences(test['clean_text']), maxlen=MAX_SEQUENCE_LENGTH)
print(train_seq.shape)

print(val_seq.shape)

print(test_seq.shape)
def build_model():

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH, ))

    x = Embedding(nb_words, EMBEDDING_DIM + 2, weights=[embedding_matrix],trainable = True)(sequence_input)

    x = SpatialDropout1D(0.2)(x)

    x = Bidirectional(GRU(128, return_sequences=True,dropout=0.1,recurrent_dropout=0.1))(x)

    x = Conv1D(64, kernel_size = 3, padding = "valid", kernel_initializer = "glorot_uniform")(x)

    avg_pool = GlobalAveragePooling1D()(x)

    max_pool = GlobalMaxPooling1D()(x)

    x = concatenate([avg_pool, max_pool]) 



    dropout = SpatialDropout1D(0.2)



    dense_input = Input(shape=(100,))

    

    dense_vector = BatchNormalization()(dense_input)

    

    

    feature_vector = concatenate([x, dense_vector])



    feature_vector = Dense(128, activation="relu")(feature_vector)

    

    output = Dense(6, activation="sigmoid")(feature_vector)

    

    model = Model(inputs=[sequence_input, dense_input], outputs=output)

    return model
print(x_train_vec.shape)

print(x_val_vec.shape)

print(x_test_tfidf_vec.shape)



print(train_seq.shape)

print(val_seq.shape)

print(test_seq.shape)



print(y_train.shape)

print(y_val.shape)
model = build_model()

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

early_stopping = EarlyStopping(monitor="val_acc", patience=2, verbose=1)



print("Training the model")

model.fit([train_seq, x_train_vec], y_train.values, validation_data=([val_seq, x_val_vec], y_val.values),epochs=2,

          batch_size=1024, shuffle=True, callbacks=[early_stopping], verbose=1)
model.summary()
SVG(model_to_dot(model).create(prog='dot', format='svg'))
test_preds = model.predict([test_seq, x_test_tfidf_vec], batch_size=1024, verbose=1)
test_preds.shape
labels = ['toxic','severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
y_df = np.where(test_preds > 0.5, 1, 0)



y_df = pd.DataFrame(y_df, columns=['toxic','severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'])

y_df = y_df.astype('int')
def get_metri_scores(y_test, y_test_pred):

    vals = precision_recall_fscore_support(y_test, y_test_pred, average='macro')

    precision = vals[0]

    recall = vals[1]

    f1 = vals[2]

    acc = accuracy_score(y_test, y_test_pred)

    return precision, recall, f1, acc
results_cv = pd.DataFrame({'labels': labels})

results_cv['acc'] = 0

results_cv['f1'] = 0

results_cv['precision'] = 0

results_cv['recall']  = 0

for col in labels:

    print(col)

    precision, recall, f1, acc = get_metri_scores(test[col], y_df[col])

    results_cv['acc'][results_cv['labels']==col] = acc

    results_cv['f1'][results_cv['labels']==col] = f1

    results_cv['precision'][results_cv['labels']==col] = precision

    results_cv['recall'][results_cv['labels']==col] = recall
results_cv