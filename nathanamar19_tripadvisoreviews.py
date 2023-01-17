import pandas as pd
from nltk.tag import UnigramTagger
from nltk.corpus import treebank
import nltk
import re
import matplotlib.pyplot as plt

replacement_patterns = [
    (r'won\'t', 'will not'),
    (r'can\'t', 'cannot'),
    (r'i\'m', 'i am'),
    (r'ain\'t', 'is not'),
    (r'(\w+)\'ll', '\g<1> will'),
    (r'(\w+)n\'t', '\g<1> not'),
    (r'(\w+)\'ve', '\g<1> have'),
    (r'(\w+)\'s', '\g<1> is'),
    (r'(\w+)\'re', '\g<1> are'),
    (r'(\w+)\'d', '\g<1> would'),
]

class RegexpReplacer(object):
    def __init__(self, patterns=replacement_patterns): 
        self.patterns = [(re.compile(regex), repl) for (regex, repl) in patterns]
    def replace(self, text):
        s = text
        for (pattern, repl) in self.patterns:
            s = re.sub(pattern, repl, s) 
        return s

replacer=RegexpReplacer()
replacer.replace("Don't hesistate to ask questions")

from sklearn.model_selection import train_test_split

import math
import random
from collections import defaultdict
from pprint import pprint
from collections import Counter
from nltk.corpus import stopwords
import re
import string
from sklearn.metrics import mean_squared_error
import nltk

# Prevent future/deprecation warnings from showing in output
import warnings
warnings.filterwarnings(action='ignore')

import numpy as np
import pandas as pd


import matplotlib.pyplot as plt
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import sentiwordnet as swn
from bs4 import BeautifulSoup             
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.corpus import stopwords 
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve

#Visualization packages
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error
import random

import nltk
nltk.download('treebank')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
#file_id='1DiCkP6qwCxIPK2TuK47US_QTVEminv5T'
#link='https://drive.google.com/uc?export=download&id={FILE_ID}'
#csv_url=link.format(FILE_ID=file_id)

#original_dataset = pd.read_csv(csv_url, sep=';', index_col='Unnamed: 0')

column_names = ['reviews.rating','reviews.text']
original_dataset = pd.read_csv('http://christophe-rodrigues.fr/eval_reviews.csv', usecols=column_names, sep=";")


original_dataset.head()
#Dimension of the dataset
original_dataset.shape
original_dataset.describe()
original_dataset.isna().sum()
original_dataset = original_dataset[original_dataset['reviews.text']!='MoreMore']
original_dataset.shape
original_dataset['reviews.rating'].value_counts()
original_dataset['reviews.rating'].value_counts().plot.bar(color='blue')
 
def preprocess_text(test):

    #Convert the text to lowercase
    test = test.lower()

    #Removing Numbers
    test=re.sub(r'\d+','',test)


    
    #Removing white spaces
    test=test.strip()
    
    #Replacer replace
    text_replaced = replacer.replace(test)
    

    
    #Tokenize
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(text_replaced)

    #Tokenize words
    from nltk.tokenize import RegexpTokenizer
    tokenizer=RegexpTokenizer("[\w]+")

    for i in range(len(sentences)):
        sentences[i] = tokenizer.tokenize(sentences[i])

    #Remove stop words

    from nltk.corpus import stopwords
    stops=set(stopwords.words('english'))

    for i in range(len(sentences)):
        sentences[i] = [word for word in sentences[i] if word not in stops]

    #Lemmatize

    from nltk.stem import WordNetLemmatizer
    lemmatizer_output=WordNetLemmatizer()

    for i in range(len(sentences)):
        for j in range(len(sentences[i])):
            sentences[i][j] = lemmatizer_output.lemmatize(sentences[i][j])


    #Join the words back into a sentence.
    a=[' '.join(s) for s in sentences]
    b=['. '.join(a)]

    return b 

review_clean = [preprocess_text(doc) for doc in original_dataset['reviews.text']]
sentences = [' '.join(r) for r in review_clean]

original_dataset['text_cleaned']=sentences
original_dataset.head()
dataset = original_dataset.copy()
dataset[dataset['reviews.rating'] != 3]
dataset['labels'] = np.where(dataset['reviews.rating'] > 2, 1, 0)
dataset.head()
dataset['labels'].value_counts()
dataset['labels'].value_counts().plot.bar(color='green')
from sklearn.model_selection import train_test_split

X = dataset.text_cleaned
y = dataset.labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(max_features=15000, binary=True)

X_train_vect = vectorizer.fit_transform(X_train)
#Utilisation de smote pour les dataset déséquilibrés
from imblearn.over_sampling import SMOTE

sm = SMOTE()

X_train_res, y_train_res = sm.fit_sample(X_train_vect, y_train)
X_test_vect = vectorizer.transform(X_test)
from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()

nb.fit(X_train_res, y_train_res)

y_pred = nb.predict(X_test_vect)
print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
print(nb.predict(vectorizer.transform(['this hotel was amazing'])))
print(nb.predict(vectorizer.transform(['This hotel was a fucking joke, have you ever seen a housekipper that doesn\'t clean room? '])))
original_dataset.head()
x1 = original_dataset['text_cleaned']
y1 = original_dataset['reviews.rating']
vect = TfidfVectorizer(ngram_range = (1,2))
x_vect1 = vect.fit_transform(x1)
x_train_c, x_test_c, y_train_c, y_test_c = train_test_split(x_vect1, y1, test_size=0.15, random_state = 10, shuffle=True)
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
lin_svc_mod = LinearSVC(C=0.13, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=0, tol=1e-05, verbose=0)
lin_svc_mod.fit(x_train_c, y_train_c)
pred = lin_svc_mod.predict(x_test_c)
print("Linear SVC:",accuracy_score(y_test_c, pred))
print("MSE: ",mean_squared_error(y_test_c,pred))

print(lin_svc_mod.predict(vect.transform(['this hotel was horrible'])))

print("Score supposé : 4")
print("Score predit : ")
print(lin_svc_mod.predict(vect.transform(['loved 	stayed warwick overnight getway enjoy christmas shopping 	warwick exceeded expectations 	staff wonderful extrememly friendly room clean service lounge wonderful 	came contact hotel friendly 	women bathroom lever lounge well.. think haunted totally creepy vibe lights anywho 	really enjoyed stay going couple days 	 '])))

from sklearn.ensemble import RandomForestClassifier
rmfr = RandomForestClassifier()
rmfr.fit(x_train_c, y_train_c)
predrmfr = rmfr.predict(x_test_c)
print("Score:",round(accuracy_score(y_test_c,predrmfr)*100,2))
print("MSE: ",mean_squared_error(y_test_c,predrmfr))


parameters = {
    "n_estimators":[5,10,50,100,250],
    "max_depth":[2,4,8,16,32,64]
    
}
from sklearn.model_selection import GridSearchCV
cv = GridSearchCV(rmfr,parameters,cv=5)
cv.fit(x_train_c, y_train_c)
def display(results):
    print(f'Best parameters are: {results.best_params_}')
    print("\n")
    mean_score = results.cv_results_['mean_test_score']
    std_score = results.cv_results_['std_test_score']
    params = results.cv_results_['params']
    for mean,std,params in zip(mean_score,std_score,params):
        print(f'{round(mean,3)} + or -{round(std,3)} for the {params}')
display(cv)

from sklearn.ensemble import RandomForestClassifier
rmfrclass = RandomForestClassifier(max_depth = 64,n_estimators = 10 )
rmfrclass.fit(x_train_c, y_train_c)
predrmfrclass = rmfrclass.predict(x_test_c)
print("Score:",round(accuracy_score(y_test_c,predrmfrclass)*100,2))
print("MSE: ",mean_squared_error(y_test_c,predrmfrclass))

from sklearn.svm import SVC
svm = SVC(random_state=101)
svm.fit(x_train_c,y_train_c)
predsvm = svm.predict(x_test_c)
print("Score:",round(accuracy_score(y_test_c,predsvm)*100,2))
print("MSE: ",mean_squared_error(y_test_c,predsvm))
import numpy as np
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train_c,y_train_c)
print("Score: ", reg.score(x_test_c, y_test_c))
pred_lin_reg = reg.predict(x_test_c)
print("MSE: ",mean_squared_error(y_test_c,pred_lin_reg))
#Supposé 2
print(reg.predict(vect.transform([' 	1st time seattle delayed anniversary trip wanted stay nicer hotels room reminded holiday inn level hotel 	plain room extra pillows 	bathroom ordinary corian sink ordinary bathroom 	room higher floor looking freeway loud 	reason earplugs sleep cd 	asked switch rooms told probably stay way stay 2 nights staying hotel different area town 	luggage room decided eat 	stopped concierge asked good place walk rudely told just walk area 	not sure concierge doorman just sitting desk expected help 	decided night hotel come day earlier happily said 	used club points crowne rooms maybe lousy experience opted leave pay room luxury hotel hotel 1000'])))
#from sklearn import datasets,linear_model
#from sklearn.model_selection import GridSearchCV
#parameters = {'kernel':('linear', 'rbf')}
#svc=linear_model.ARDRegression(n_iter=300,tol=0.001)
#clf = GridSearchCV(svc, parameters, cv=5)
#clf
import keras
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from keras import Sequential
from keras.layers import Embedding
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Lambda
from keras.layers import LSTM
import keras.backend as K

import nltk
nltk.download('treebank')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
def create_dataset(num_words, max_text_len, data):
  res = []
  for i in tqdm(range(len(data))):
    res.append([preprocess_text(data.iloc[i]["reviews.text"]), data.iloc[i]["reviews.rating"]])
  inp, targ = zip(*res)
  print("Tokenizing the data ...")
  tokenizer = keras.preprocessing.text.Tokenizer(num_words = num_words)
  tokenizer.fit_on_texts([i[0] for i in inp])
  tensor = [ tokenizer.texts_to_sequences(i)[0] for i in inp]
  tensor = keras.preprocessing.sequence.pad_sequences(tensor, padding='post',   value=0, maxlen=max_text_len)
  print("Splitting the data into train/val datasets (0.2) ...")
  input_tensor_train, input_tensor_test, target_tensor_train, target_tensor_test = train_test_split(tensor, targ, test_size=0.1)

  return (input_tensor_train, np.array(target_tensor_train)-1), (input_tensor_test, target_tensor_test),  tokenizer
vocab_size = 5000
embed_size = 300
max_text_len = 200
learning_rate = 0.001
batch_size = 128
n_epochs = 4
train_set, test_set, dictionary = create_dataset(vocab_size, max_text_len, original_dataset)
model = Sequential()
model.add(Embedding(vocab_size, embed_size, input_shape=(max_text_len,), mask_zero=True))
model.add(LSTM(512, return_sequences=True))
model.add(LSTM(512, return_sequences=True))
model.add(Lambda(lambda x : K.mean(x, axis=1)))
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(100, activation="relu"))
model.add(Dropout(0.2))
#model.add(Dense(50, activation="relu"))
model.add(Dense(5, activation="softmax"))
import tensorflow as tf
# Variable-length int sequences.
query_input = tf.keras.Input(shape=(max_text_len,), dtype='int32')

# Embedding lookup.
token_embedding = tf.keras.layers.Embedding(vocab_size, embed_size)
# Query embeddings of shape [batch_size, Tq, dimension].
query_embeddings = token_embedding(query_input)
# Value embeddings of shape [batch_size, Tv, dimension].
value_embeddings = token_embedding(query_input)

# CNN layer.
cnn_layer = tf.keras.layers.Conv1D(
    filters=100,
    kernel_size=4,
    # Use 'same' padding so outputs have the same shape as inputs.
    padding='same')
# Query encoding of shape [batch_size, Tq, filters].
query_seq_encoding = cnn_layer(query_embeddings)
# Value encoding of shape [batch_size, Tv, filters].
value_seq_encoding = cnn_layer(value_embeddings)

# Query-value attention of shape [batch_size, Tq, filters].
query_value_attention_seq = tf.keras.layers.AdditiveAttention()(
    [query_seq_encoding, value_seq_encoding])

# Reduce over the sequence axis to produce encodings of shape
# [batch_size, filters].
query_encoding = tf.keras.layers.GlobalAveragePooling1D()(
    query_seq_encoding)
query_value_attention = tf.keras.layers.GlobalAveragePooling1D()(
    query_value_attention_seq)

fcn1 = tf.keras.layers.Dense(256)(query_value_attention)
dropout1 = tf.keras.layers.Dropout(0.5)(fcn1)
fcn2 = tf.keras.layers.Dense(100)(dropout1)
dropout2 = tf.keras.layers.Dropout(0.2)(fcn2)

output = tf.keras.layers.Dense(5, activation="softmax")(dropout2)

model = tf.keras.Model(query_input, output)
model.summary()
optimizer = tf.keras.optimizers.Adam()
model.compile(loss="sparse_categorical_crossentropy", optimizer = optimizer, metrics=["accuracy"])