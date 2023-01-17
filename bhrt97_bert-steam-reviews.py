import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import scipy
import matplotlib.pyplot as plt
import scikitplot as skplt
from wordcloud import WordCloud
from bs4 import BeautifulSoup
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
!wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow_hub as hub
import tokenization
import os
import re
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/nlphack/dataset/train.csv')
game_df = pd.read_csv('/kaggle/input/nlphack/dataset/game_overview.csv')
testing_df=  pd.read_csv('/kaggle/input/nlphack/dataset/test.csv')
df.head(5)
def rep(text):
    text = re.sub('♥♥♥♥',"",text)
    return text

df['user_review']=df.user_review.apply(rep)
testing_df['user_review']=testing_df.user_review.apply(rep)
def low(text):
    return text.lower()

df['user_review']=df.user_review.apply(low)
testing_df['user_review']=testing_df.user_review.apply(low)

# testing_df.head(5)
def asc(text):
    text = re.sub('[^a-zA-Z]'," ",text)
    return text

df['user_review']=df.user_review.apply(asc)
testing_df['user_review']=testing_df.user_review.apply(asc)


# testing_df.head(5)
def space(text):
#     text = re.sub(' +', ' ', text)
    text = " ".join(text.split())
    return text

df['user_review']=df.user_review.apply(space)
testing_df['user_review']=testing_df.user_review.apply(space)
def clean(text):
    text = re.sub(' ll', 'will', text)
    text = re.sub('lvl', 'level', text)
    text = re.sub('dev', 'developer', text)
#     text = re.sub('ll', 'will', text)

    return text

df['user_review']=df.user_review.apply(clean)
testing_df['user_review']=testing_df.user_review.apply(clean)
testing_df['user_review'][2445]
testing_df.drop(["review_id","title","year"],axis=1,inplace=True)
testing_df.head(5)
df.drop(['review_id'],axis=1,inplace=True)
result = pd.merge(df, game_df,on='title', how='left')
result.drop(['year'],axis=1,inplace=True)
result.tail(5)
result.isnull().any()
result.drop(["overview","developer","publisher"],axis=1,inplace=True)
result.head(2)
testing_df.sample(2)
def punctuation_removal(messy_str):
    clean_list = [char for char in messy_str if char not in string.punctuation]
    clean_str = ''.join(clean_list)
    return clean_str
result['user_review'] = result['user_review'].apply(punctuation_removal)
testing_df['user_review'] = testing_df['user_review'].apply(punctuation_removal)


from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
stop = stopwords.words('english')
print(stop)
stop.remove('no')
stop.remove('not')
stop.remove("don't")

stop_words = []

for item in stop: 
    new_item = punctuation_removal(item)
    stop_words.append(new_item) 
print(stop_words)
def stopwords_removal(messy_str):
    messy_str = word_tokenize(messy_str)
    return [word.lower() for word in messy_str 
            if word.lower() not in stop_words ]
result['user_review'] = result['user_review'].apply(stopwords_removal)
testing_df['user_review'] = testing_df['user_review'].apply(stopwords_removal)

import re
def drop_numbers(list_text):
    list_text_new = []
    for i in list_text:
        if not re.search('\d', i):
            list_text_new.append(i)
    return ' '.join(list_text_new)
result['user_review'] = result['user_review'].apply(drop_numbers)
testing_df['user_review'] = testing_df['user_review'].apply(drop_numbers)

print(result.user_review.str.len().mean())
print(testing_df.user_review.str.len().mean())
X = result.drop('user_suggestion',axis=1)
y = result['user_suggestion']
X.user_review.str.len().mean()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state = 42, test_size=0.25)
print('\n train X: {} \n train y: {} \n Val X: {} \n val y: {}'.format((X_train.shape),(y_train.shape),(X_test.shape),(y_test.shape)))
X_train.user_review.str.len().mean()
X_test.user_review.str.len().mean()
module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"
bert_layer = hub.KerasLayer(module_url, trainable=True)
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
X_train_bert = X_train.user_review
X_test_bert = X_test.user_review


def bert_encode(input_text, tokenizer, max_len = 512):
    token_input = [] 
    mask_input = []
    seg_input = []
    
    for text in input_text:
        text = tokenizer.tokenize(text)
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)
        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence)      
        token_input.append(tokens + [0]*pad_len)
        mask_input.append([1]*len(input_sequence) + [0]*pad_len)
        seg_input.append([0] * max_len)
        
    return np.array(token_input), np.array(mask_input), np.array(seg_input)
def build_model(bert_layer, max_len = 512):
    input_word_ids = Input(shape=(max_len, ),dtype = tf.int32,name = 'input_words_ids')
    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")
    
    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    clf_output = sequence_output[:, 0, :]
    out = Dense(1, activation='sigmoid')(clf_output)
    
    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    model.compile(Adam(lr=2e-6), loss='binary_crossentropy', metrics=['accuracy'])   #0.000 002
    
    return model
train_input = bert_encode(X_train_bert.values, tokenizer, max_len=240)
test_input = bert_encode(X_test_bert.values, tokenizer, max_len=240)
train_labels = y_train.values
testing_input = bert_encode(testing_df.user_review, tokenizer, max_len=240)
model = build_model(bert_layer, max_len=240)
model.summary()
train_history = model.fit(
    train_input, train_labels,
    validation_split=0.2,
    epochs=2,
    batch_size=10
)
prediction = model.predict(test_input)
preds = []
for x in prediction:
    preds.append(int(x.round()))

from sklearn.metrics import accuracy_score
print("Accuracy: ", accuracy_score(y_test.values,preds))
from sklearn.metrics import confusion_matrix 

print(confusion_matrix(y_test.values, preds))
preds.count(1)
preds.count(0)
labels = [0, 1]
cm = confusion_matrix(y_test.values, preds, labels)
print(cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
from sklearn.metrics import classification_report 
print(classification_report(y_test.values, preds))
pred_result = prediction = model.predict(testing_input)
submission = pred_result.round().astype(int)
submission=pd.DataFrame(submission)

sub = pd.read_csv('/kaggle/input/steam-recommendation-nlp-dataset/test.csv')
submission['review_id']=sub['review_id']

submission= submission[['review_id',0]]
submission.rename(columns = {0:'user_suggestion'}, inplace = True) 

submission.head(4)


submission.to_csv('submission_240stop_clean.csv', index=False)