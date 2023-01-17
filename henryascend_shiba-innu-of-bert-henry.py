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
!pip install bert-for-tf2

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
import bert

# Loading pretrained bert layer
BertTokenizer = bert.bert_tokenization.FullTokenizer
bert_layer_original = hub.KerasLayer( "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2",
                            trainable=True)

# load the dataset

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

sample_submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")
train_df = pd.read_csv("../input/nlp-getting-started/train.csv")
test_df = pd.read_csv("../input/nlp-getting-started/test.csv")
correct_submission = pd.read_csv("../input/correct-submission/correct_submission.csv")

# text preprosessing

from nltk.stem import PorterStemmer #normalize word form
from nltk.probability import FreqDist #frequency word count
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords #stop words
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.probability import FreqDist 

def text_cleaning_hyperlink(text,rep):
    
    #remove hyper link
    return re.sub(r"http\S+","{}".format(rep),text) #remove hyperlink


def text_cleaning_punctuation(text):
    defined_punctuation = string.punctuation.replace('#','')
    #defined_punctuation = string.punctuation.replace('@','')
    translator = str.maketrans( defined_punctuation, ' '*len( defined_punctuation)) #remove punctuation    
    return text.translate(translator)


def text_cleaning_stopwords(text):
    
    stop_words = set(stopwords.words('english'))
    word_token = word_tokenize(text)
    filtered_sentence = [w for w in word_token if not w in stop_words]
    return ' '.join(filtered_sentence) #return string of no stopwords


# remove digits from the text

def remove_digits(txt):
    
    no_digits = ''.join(i for i in txt if not i.isdigit())
    return no_digits

def text_clean_username(txt,rep):
   
    return re.sub(r"@\S+","{}".format(''),txt) #remove hyperlink
    return no_username
    


def text_clean(tweet):
    
    # Special characters
    tweet = re.sub(r"\x89Û_", "", tweet)
    tweet = re.sub(r"\x89ÛÒ", "", tweet)
    tweet = re.sub(r"\x89ÛÓ", "", tweet)
    tweet = re.sub(r"\x89ÛÏ", "", tweet)
    tweet = re.sub(r"\x89Û÷", "", tweet)
    tweet = re.sub(r"\x89Ûª", "", tweet)
    tweet = re.sub(r"\x89Û\x9d", "", tweet)
    tweet = re.sub(r"å_", "", tweet)
    tweet = re.sub(r"\x89Û¢", "", tweet)
    tweet = re.sub(r"\x89Û¢åÊ", "", tweet)
    tweet = re.sub(r"åÊ", "", tweet)
    tweet = re.sub(r"åÈ", "", tweet)
    
#     tweet = re.sub(r'\n',' ', tweet) # Remove line breaks
#     tweet = re.sub('\s+', ' ', tweet).strip() # Remove leading, trailing, and extra spaces
    
    return tweet

def clean( tweet ):
    
    tweet =  text_clean(tweet)
    tweet =  remove_digits(tweet)
    
    tweet =  text_cleaning_hyperlink(tweet, 'http')
    tweet =  text_cleaning_punctuation(tweet)
    #tweet =  text_clean_username(tweet,'@')
    #tweet =  text_cleaning_stopwords(tweet)
    return tweet


def find_hashtags(tweet):
    return " ".join([match.group(0)[1:] for match in re.finditer(r"#\w+", tweet)]) or 'no'

def find_mentions(tweet):
    return " ".join([match.group(0)[1:] for match in re.finditer(r"@\w+", tweet)]) or 'no'

def find_links(tweet):
    return " ".join([match.group(0)[:] for match in re.finditer(r"https?://\S+", tweet)]) or 'no'




import re
import string



train_df['text_clean'] = train_df['text'].apply(lambda s : clean(s))
train_df['hashtags'] = train_df['text'].apply(lambda x: find_hashtags(x))
train_df['mentions'] = train_df['text'].apply(lambda x: find_mentions(x))
train_df['links'] = train_df['text'].apply(lambda x: find_links(x))

test_df['text_clean'] = test_df['text'].apply(lambda s : clean(s))
test_df['hashtags'] = test_df['text'].apply(lambda x: find_hashtags(x))
test_df['mentions'] = test_df['text'].apply(lambda x: find_mentions(x))
test_df['links'] = test_df['text'].apply(lambda x: find_links(x))
    
bert_layer = bert_layer_original

# Loading tokenizer from the bert layer

vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()

tokenizer = BertTokenizer(vocab_file, do_lower_case)
# Test and show an example of tokens
idx = 5
text = train_df.text_clean[idx]

print(train_df.text_clean[idx])
print(train_df.hashtags[idx])
print(train_df.mentions[idx])
print(train_df.links[idx])


#tokenize 
tokens_list = tokenizer.tokenize(text)
print("Text after tokenization like this:", tokens_list)

#initialize dimension
max_len = 25
text = tokens_list[:max_len-2]
input_sequence = ["[CLS]"] + text +["[SEP]"]
print("Text after adding flag -[ClS] and [SEP]: ", input_sequence )


tokens = tokenizer.convert_tokens_to_ids(input_sequence)
print("tokens to their id: ", tokens)

pad_len = max_len  - len(input_sequence)
tokens += [0] * pad_len
print("tokens: ", tokens)

print(pad_len)
pad_masks = [1] * len(input_sequence) + [0] * pad_len
print("Pad Masking: ", pad_masks)

segment_ids = [0] * max_len
print("Segment Ids: ",segment_ids)
# Function to encoe the text into tokens, mask, and segment flags

import numpy as np


def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        text = tokenizer.tokenize(text)
            
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)
        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
    
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)

MAX_LEN = 160


from tensorflow.keras.layers import Input



input_word_ids = Input(shape=(MAX_LEN,), dtype=tf.int32, name="input_word_ids")
input_mask = Input(shape=(MAX_LEN,), dtype=tf.int32, name="input_mask")
segment_ids = Input(shape=(MAX_LEN,), dtype=tf.int32, name="segment_ids")

    
#output

from tensorflow.keras.layers import Dense

_, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
clf_output = sequence_output[:, 0, :]
out1 = Dense(128, activation='relu')(clf_output)
out = Dense(1, activation='sigmoid')(out1)

#model initialization 

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD

model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)


from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))




#optimizer = SGD(learning_rate=0.0001, momentum=0.8)
optimizer = Adam(lr=6e-6)
model.compile(optimizer = optimizer, loss='binary_crossentropy', metrics=['accuracy',f1_m,precision_m, recall_m])

model.summary()


from sklearn.model_selection import train_test_split


##individual text
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(train_df['text_clean'], train_df.target, test_size=0.3)
train_input = bert_encode(X_train_1, tokenizer, max_len=MAX_LEN)
test_input = bert_encode(X_test_1, tokenizer, max_len=MAX_LEN)
train_labels = y_train_1
test_labels = y_test_1

##submision text

# train_input = bert_encode(train_df['text_clean'], tokenizer, max_len=MAX_LEN)
# test_input = bert_encode(test_df['text_clean'], tokenizer, max_len=MAX_LEN)
# train_labels = train_df.target.values


train_history = model.fit(
    train_input, train_labels,
    validation_split=0.2,
    epochs=10,
    batch_size=16
)

model.save('model.h5')



from sklearn.metrics import f1_score

test_pred = model.predict(test_input)
preds = test_pred.round().astype(int)




# evaluate the model
loss, accuracy, f1_score, precision, recall = model.evaluate(test_input, test_labels, verbose=1)
#loss, f1_score = model.evaluate(test_input, test_labels, verbose=1)
print("Accuracy:", accuracy)
print("f1_score:", f1_score)




# test_input_official = bert_encode(test_df['text_clean'], tokenizer, max_len=MAX_LEN)
# test_pred = model.predict(test_input_official)
# preds = test_pred.round().astype(int)


# sample_submission["target"] = preds
# sample_submission.target.value_counts()
# sample_submission.to_csv("submission.csv", index = False)



# Showing Confusion Matrix
# thanks to https://www.kaggle.com/marcovasquez/basic-nlp-with-tensorflow-and-wordcloud

from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def plot_cm(y_true, y_pred, title, figsize=(5,5)):
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    plt.title(title)
    sns.heatmap(cm, cmap= "YlGnBu", annot=annot, fmt='', ax=ax)
plot_cm(preds, test_labels, 'Confusion matrix for BERT model', figsize=(7,7))
import itertools
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion Matrix',
                          cmap=plt.cm.spring):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=26)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", 
                 color="white" if cm[i, j] < thresh else "black", fontsize=40)
    
    plt.tight_layout()
    plt.ylabel('True labels', fontsize=24)
    plt.xlabel('Predicted labels', fontsize=24)

    return plt
test_input_official = bert_encode(test_df['text_clean'], tokenizer, max_len=MAX_LEN)
test_pred = model.predict(test_input_official)
preds = test_pred.round().astype(int)

correct_labels = correct_submission.target

loss, accuracy, f1_score, precision, recall = model.evaluate(test_input_official, correct_labels, verbose=1)

cm = confusion_matrix(correct_labels ,preds )
fig = plt.figure(figsize=(8, 8))
plot = plot_confusion_matrix(cm, classes=['Irrelevant','Disaster'], normalize=False, title='Confusion matrix\n for BERT')
plt.show()

sample_submission["target"] = preds
sample_submission.target.value_counts()
sample_submission.to_csv("submission.csv", index = False)
from sklearn.metrics import roc_curve
import seaborn as sns

# predict_proba gives the probabilities for the target (0 and 1 in our case) as a list (array).
# The number of probabilities for each row is equal to the number of categories in target variable (2 in our case).

probabilities_LSTM = model.predict(test_input_official)

print(probabilities_LSTM)


# Using [:,1] gives us the probabilities of getting the output as 1

probability_of_ones_LSTM = probabilities_LSTM
# probability_of_ones_LogisticRegression = probabilities_LogisticRegression[:,1]


# roc_curve returns:
# - false positive rates (FPrates), i.e., the false positive rate of predictions with score >= thresholds[i]
# - true positive rates (TPrates), i.e., the true positive rate of predictions with score >= thresholds[i]
# - thresholds

FPrates_LSTM, TPrates_LSTM, thresholds_LSTM = roc_curve(correct_labels, probability_of_ones_LSTM)
# FPrates_LogisticRegression, TPrates_LogisticRegression, thresholds_LogisticRegression = roc_curve(y_test, probability_of_ones_LogisticRegression)



# plotting the ROC Curve to visualize all the methods

sns.set_style('whitegrid')
plt.figure(figsize = (10, 8))

plt.plot(FPrates_LSTM, TPrates_LSTM, label = 'BERT')
# plt.plot(FPrates_LogisticRegression, TPrates_LogisticRegression, label = 'Logistic Regression')


plt.plot([0, 1], [0, 1], color = 'green', linestyle = '--')
plt.axis([0, 1, 0, 1])
plt.xlabel('False Positive Rate', fontsize = 14)
plt.ylabel('True Positive Rate', fontsize = 14)
plt.title('ROC Curve', fontsize = 14)
plt.legend(loc = "lower right")
plt.show()
