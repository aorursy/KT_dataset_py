import csv

import pandas as pd

import numpy as np

import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

from sklearn import metrics

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB

from os import path

from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
URL_Train ='https://raw.githubusercontent.com/cacoderquan/Sentiment-Analysis-on-the-Rotten-Tomatoes-movie-review-dataset/master/train.tsv'

URL_Test ='https://raw.githubusercontent.com/cacoderquan/Sentiment-Analysis-on-the-Rotten-Tomatoes-movie-review-dataset/master/test.tsv'
train = pd.read_csv(URL_Train,sep='\t')

test = pd.read_csv(URL_Test,sep='\t')

train.head()
train['Sentiment'].value_counts()
# Put all the fraud class in a separate dataset.

fraud_df1 = train.loc[train['Sentiment'] == 1].sample(n=7072,random_state=42)

fraud_df2 = train.loc[train['Sentiment'] == 2].sample(n=7072,random_state=42)

fraud_df3 = train.loc[train['Sentiment'] == 3].sample(n=7072,random_state=42)

fraud_df4 = train.loc[train['Sentiment'] == 4].sample(n=7072,random_state=42)



#Randomly select 492 observations from the non-fraud (majority class)

non_fraud_df = train[train['Sentiment'] == 0]



# Concatenate both dataframes again

normalized_df = pd.concat([non_fraud_df, fraud_df1,fraud_df2,fraud_df3,fraud_df4])



#plot the dataset after the undersampling

#plt.figure(figsize=(8, 8))

#sns.countplot('Sentiment', data=normalized_df)

#plt.title('Balanced Classes')

#plt.show()

normalized_df.head()

normalized_df.shape
#from imblearn.over_sampling import SMOTE

#sm = SMOTE(ratio='minority', random_state=7)



#oversampled_trainX, oversampled_trainY = sm.fit_sample(train.drop('Sentiment', axis=1), train['Sentiment'])

#oversampled_train = pd.concat([pd.DataFrame(oversampled_trainY), pd.DataFrame(oversampled_trainX)], axis=1)

#oversampled_train.columns = train.columns





#from imblearn.over_sampling import SMOTENC

#smote_nc = SMOTENC(categorical_features=[0, 2], random_state=0)

#X_resampled, y_resampled = smote_nc.fit_resample(train.drop('Sentiment', axis=1), train['Sentiment'])
test.head()

#test['Phrase'][4]
print(train.shape,"\n",test.shape)
print("\t",train.isnull().values.any(), "\n\t",

      test.isnull().values.any()

     )
#sanitization

fullSent_train = normalized_df



fullSent_train.head()

#fullSent['Phrase'][156]
#sanitization

fullSent_test = test



fullSent_test.head()

#fullSent['Phrase'][156]



fullSent_test.shape
print (len(train.groupby('SentenceId').nunique()),

      len(test.groupby('SentenceId').nunique())

      )
senti = train.groupby(["Sentiment"]).size()

senti = senti / senti.sum()

fig, ax = plt.subplots(figsize=(8,8))

sns.barplot(senti.keys(), senti.values);
StopWords = ENGLISH_STOP_WORDS

print(StopWords)
text = " ".join(review for review in train.Phrase)

print ("There are {} words in the combination of all review.".format(len(text)))
# Create stopword list:

stopwords = set(StopWords)

stopwords.update(["drink", "now", "wine", "flavor", "flavors"])



# Generate a word cloud image

wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)



# Display the generated image:

# the matplotlib way:

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
BOW_Vectorizer = CountVectorizer(strip_accents='unicode',

                                 stop_words=StopWords,

                                 ngram_range=(1,3),

                                 analyzer='word',

                                 min_df=5,

                                 max_df=0.5)



#BOW_Vectorizer.fit(list(fullSent['Phrase']))



#create tfidf vectorizer 

tfidf_vectorizer = TfidfVectorizer(min_df=5,

                                 max_df=5,

                                  analyzer='word',

                                  strip_accents='unicode',

                                  ngram_range=(1,3),

                                  sublinear_tf=True,

                                  smooth_idf=True,

                                  use_idf=True,

                                  stop_words=StopWords)



tfidf_vectorizer.fit(list(fullSent_train['Phrase']))
tfidf_vectorizer.fit(list(fullSent_test['Phrase']))
X_train = fullSent_train['Phrase']

Y_train = fullSent_train['Sentiment']

X_test = fullSent_test['Phrase']

X_train.shape, Y_train.shape, X_test.shape
#build train and test datasets

#phrase = fullSent['Phrase']

#sentiment = fullSent['Sentiment']

#phrase[0], sentiment[0]
#X_train,X_test,Y_train,Y_test = train_test_split(phrase,sentiment,test_size=0.3,random_state=4)



#X_train.shape, Y_train.shape, X_test.shape, Y_test.shape

#calling BoW Model and transforming to spase matrix

train_bow=tfidf_vectorizer.transform(X_train)

test_bow=tfidf_vectorizer.transform(X_test)

train_bow.shape[1]

bow_fea_vec_train = pd.DataFrame(train_bow.toarray(), columns = tfidf_vectorizer.get_feature_names())

bow_fea_vec_train.head(5)



bow_fea_vec_test = pd.DataFrame(test_bow.toarray(), columns = tfidf_vectorizer.get_feature_names())

bow_fea_vec_test.head(5)
from keras import backend as K

def recall_m(y_true, y_pred):

  true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

  possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

  recall = true_positives / (possible_positives + K.epsilon())

  return recall



def precision_m(y_true, y_pred):

  true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

  predicted_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

  precision = true_positives / (predicted_positives + K.epsilon())

  return precision



def f1_m(y_true, y_pred):

  precision = precision_m(y_true, y_pred)

  recall = recall_m(y_true, y_pred)

  return 2*((precision*recall)/(precision+recall+K.epsilon()))
from keras.models import Sequential

from keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Flatten

from keras.layers import Activation, Conv1D, GlobalMaxPooling1D

from keras import optimizers

import keras
lr = 1e-3

batch_size = 128

num_epochs = 10

decay = 1e-4

mode = "reg"

n_class = 5 #5
fea_train_dim = bow_fea_vec_train.shape[1]

print(fea_train_dim, n_class)



X_train = bow_fea_vec_train.values.reshape((bow_fea_vec_train.shape[0], bow_fea_vec_train.shape[1], 1))

X_train.shape





fea_test_dim = bow_fea_vec_test.shape[1]

print(fea_test_dim, n_class)



X_test = bow_fea_vec_test.values.reshape((bow_fea_vec_test.shape[0], bow_fea_vec_test.shape[1], 1))

X_test.shape

#Y_train



Y_train = keras.utils.to_categorical(Y_train, num_classes=None, dtype='float32')

#Y_test = keras.utils.to_categorical(X_test)

X_test.shape
def baseline_cnn_model(fea_matrix, n_class, mode, compiler):

  #create model

  model = Sequential()

  model.add(Conv1D(filters=64, kernel_size = 3, activation = 'relu',

                  input_shape=(fea_matrix.shape[1], fea_matrix.shape[2])))

  model.add(MaxPooling1D(pool_size = 2))

  model.add(Conv1D(filters=128, kernel_size = 3, activation = 'relu'))

  model.add(MaxPooling1D(pool_size=2))

  model.add(Dropout(0.25))

  model.add(Flatten())

  model.add(Dense(128, activation='relu'))

  model.add(Dropout(0.5))

  model.add(Activation('relu'))

  model.add(Dense(n_class))

  if n_class==1 and mode == "cla":

    model.add(Activation('softmax'))  

    #comoile the model

    model.compile(optimizer=compiler, loss = 'sparse_categorical_crossentropy',

                 metrics=['acc', f1_m, precision_m, recall_m])

  else:

    model.add(Activation('sigmoid'))

    # compile the model

    model.compile(optimizer=compiler, loss = 'binary_crossentropy',

            metrics=['acc', f1_m, precision_m, recall_m])

  return model

  
adm = optimizers.Adam(lr = lr, decay = decay)

sgd = optimizers.SGD(lr = lr, nesterov = True, momentum = 0.7, decay = decay)

Nadam = optimizers.Nadam(lr = lr, beta_1=0.9, beta_2=0.999, epsilon = 1e-08)

model = baseline_cnn_model(X_train, n_class, mode, Nadam)

model.summary()


acc_loss = model.fit(X_train, Y_train, batch_size = batch_size, 

          epochs = num_epochs, verbose=1, validation_split = 0.2)
Xnew=model.predict_classes(X_test)

Xnew
submission = pd.DataFrame({'PhraseId':fullSent_test['PhraseId'],'Sentiment':Xnew})

#fullSent_test['Phrase']

submission.head()
fullSent_test.shape
submission.to_csv('submission.csv', index = False)