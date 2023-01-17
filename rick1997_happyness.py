import pandas as pd

#-------------------data Visualisation import

import seaborn as sns

from tqdm import tqdm

import matplotlib.pyplot as plt

import string

#-------For Natural Language Processing data cleaning

from sklearn.feature_extraction.text import TfidfTransformer

from keras.preprocessing import sequence

from keras.preprocessing.text import Tokenizer

from nltk.tokenize import word_tokenize

from bs4 import BeautifulSoup

import re

#-------Data model

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

from keras.preprocessing import sequence

from keras.preprocessing.text import Tokenizer

from keras.layers import Dense,Dropout,Embedding,LSTM

from keras.losses import categorical_crossentropy

from keras.optimizers import Adam

from keras.models import Sequential

from sklearn.metrics import f1_score, accuracy_score

import keras.backend as K
df_test = pd.read_csv("../input/hm_test.csv")

df_train = pd.read_csv("../input/hm_train.csv")

sample_submission = pd.read_csv("../input/sample_submission.csv")
plt.figure(figsize=(12,4))

sns.countplot(data=df_train,x='predicted_category')
df_train["happytime"] = df_train["predicted_category"].map({"affection": 0, "exercise" : 1, "bonding" : 2,"leisure" : 3,"achievement" : 4,"enjoy_the_moment" : 5,"nature" : 6})
def clean_sentences(df):

    reviews=[]

    for sent in tqdm(df['cleaned_hm']):

        review_text = BeautifulSoup(sent).get_text()

        review_text = re.sub("[^a-zA-Z]"," ", review_text)

        review_text=review_text.translate(str.maketrans('', '', string.punctuation))

        words = word_tokenize(review_text.lower())

        reviews.append(words)

    return reviews
train_sentences = clean_sentences(df_train)

test_sentences =clean_sentences(df_test)

print(len(train_sentences))

print(len(test_sentences))
target=df_train.happytime.values

y_target=to_categorical(target)

num_classes=y_target.shape[1]

X_train,X_val,y_train,y_val=train_test_split(train_sentences,y_target,test_size=0.2,stratify=y_target)
unique_words = set()

len_max = 0



for sent in tqdm(X_train):

    

    unique_words.update(sent)

    

    if(len_max<len(sent)):

        len_max = len(sent)

print(len(list(unique_words)))

print(len_max)
tokenizer = Tokenizer(num_words=len(list(unique_words)))

tokenizer.fit_on_texts(list(X_train))

X_train = tokenizer.texts_to_sequences(X_train)

X_val = tokenizer.texts_to_sequences(X_val)

X_train = sequence.pad_sequences(X_train, maxlen=len_max)

X_val = sequence.pad_sequences(X_val, maxlen=len_max)

#X_test = sequence.pad_sequences(X_test, maxlen=len_max)



print(X_train.shape,X_val.shape)
X_test = tokenizer.texts_to_sequences(test_sentences)

X_test = sequence.pad_sequences(X_test, maxlen=len_max)
def get_f1(y_true, y_pred): 

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    precision = true_positives / (predicted_positives + K.epsilon())

    recall = true_positives / (possible_positives + K.epsilon())

    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())

    return f1_val
model=Sequential()

model.add(Embedding(len(list(unique_words)),300,input_length=len_max))

model.add(LSTM(128,dropout=0.5, recurrent_dropout=0.5,return_sequences=True))

model.add(LSTM(64,dropout=0.5, recurrent_dropout=0.5,return_sequences=False))

model.add(Dense(100,activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes,activation='softmax'))



model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.010),metrics=[get_f1])

model.summary()
history=model.fit(X_train, y_train, validation_data=(X_val, y_val),epochs=6, batch_size=256, verbose=1)
test_prediction=model.predict(X_test,verbose=0)

yhat_classes = model.predict_classes(X_test, verbose=0)
hmid=df_test["hmid"].values
final_answer=pd.DataFrame({'hmid':hmid,'predicted_category':yhat_classes})
final_answer["predicted_category"]=final_answer["predicted_category"].map({0:"affection",1:"exercise",

                                                                           2:"bonding",3:"leisure",

                                                                          4:"achievement",5:"enjoy_the_moment",6:"nature"})
final_answer.head()
filename = 'Predict Happiness Souce.csv'

final_answer.to_csv(filename,index=False)

print('Saved file: ' + filename)