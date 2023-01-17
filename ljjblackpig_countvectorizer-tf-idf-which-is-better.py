import numpy as np

import pandas as pd



from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV, cross_validate

from sklearn.metrics import roc_curve, auc

from sklearn.naive_bayes import MultinomialNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC



from wordcloud import WordCloud

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.utils.multiclass import unique_labels



import pandas_profiling
from pylab import rcParams

rcParams['figure.figsize'] = 8, 16
df = pd.read_csv('../input/spam.csv', encoding = 'ISO-8859-1')
pandas_profiling.ProfileReport(df)
#Creating copy of our data for latter exploration

df_v2 = df.copy()
#change the label columns to 0 for spam and 1 for ham in order to feed into the models

df_v2.loc[df_v2.v1 == 'spam', 'label'] = 0

df_v2.loc[df_v2.v1 == 'ham', 'label'] = 1

#Dropped the unamed 3 columns and rename the remaining 2 columns

df_v2 = df_v2.drop(['v1','Unnamed_2', 'Unnamed_3', 'Unnamed_4'],  axis=1)

df_v2 = df_v2.rename(index=str, columns={"v2": "text"})
#Config the setting and show

pd.set_option('display.max_colwidth', -1)

df_v2.head(3)
#Setting parameters for reproductivity

X = df_v2['text']

y = df_v2['label']

random_seed = 2019

tick_labels = ['spam', 'ham']
#Stratified train test split

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state = random_seed)
#Sanity checks no.1 train, test shape

print('The shape of the traning set is {}'.format(X_train.shape))

print('The shape of the testing set is {}'.format(X_test.shape))
#Sanity checks no.2 splited sample percentage

y_train_percentage = y_train.value_counts()[0] / y_train.value_counts()[1] * 100

y_test_percentage = y_test.value_counts()[0] / y_test.value_counts()[1] * 100

print('The percentage of spam in training y set is {:.1f}%'.format(y_train_percentage))

print('The percentage of spam in testing y set is {:.1f}%'.format(y_test_percentage))

print('which is close to the original dataset of 13.4%')
count_vectorizer = CountVectorizer(decode_error='ignore')

X_cv_train = count_vectorizer.fit_transform(X_train)

X_cv_test = count_vectorizer.transform(X_test)
#Multinomial NB

nb_model = MultinomialNB()

nb_model.fit(X_cv_train, y_train)

print("train score:", nb_model.score(X_cv_train, y_train))

print("test score:", nb_model.score(X_cv_test, y_test))
y_cv_pred = nb_model.predict(X_cv_test)

mat = confusion_matrix(y_test, y_cv_pred)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, xticklabels = tick_labels, yticklabels=tick_labels)
#Visualize the key words

#Referring to https://github.com/JJtheNOOB/machine_learning_examples/blob/master/nlp_class/spam2.py

def visualize(label):

  words = ''

  for msg in df_v2[df_v2['label'] == label]['text']:

    msg = msg.lower()

    words += msg + ' '

  wordcloud = WordCloud(width=1200, height=800, background_color="white").generate(words)

  plt.imshow(wordcloud)

  plt.axis('off')

  plt.show()
visualize(0)
visualize(1)
#Random Forest

n_estimators = range(10, 200, 20)

train_results = []

test_results = []





for estimator in n_estimators:

   rf = RandomForestClassifier(n_estimators=estimator, max_depth=40, n_jobs=-1)

   rf.fit(X_cv_train, y_train)

   train_pred = rf.predict(X_cv_train)

   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)

   roc_auc = auc(false_positive_rate, true_positive_rate)

   train_results.append(roc_auc)

   y_pred = rf.predict(X_cv_test)

   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)

   roc_auc = auc(false_positive_rate, true_positive_rate)

   test_results.append(roc_auc)





from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(n_estimators, train_results, 'b', label='Train AUC')

line2, = plt.plot(n_estimators, test_results, 'r', label='Test AUC')

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('AUC score')

plt.xlabel('n_estimators')

plt.show()
tfidf = TfidfVectorizer(decode_error='ignore')

X_tf_train = tfidf.fit_transform(X_train)

X_tf_test = tfidf.transform(X_test)
#Multinomial NB

nb_model = MultinomialNB()

nb_model.fit(X_tf_train, y_train)

print("train score:", nb_model.score(X_tf_train, y_train))

print("test score:", nb_model.score(X_tf_test, y_test))
y_tf_pred = nb_model.predict(X_tf_test)

mat = confusion_matrix(y_test, y_tf_pred)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, xticklabels = tick_labels, yticklabels=tick_labels)
#Random Forest

n_estimators = range(10, 200, 20)

train_results = []

test_results = []





for estimator in n_estimators:

   rf = RandomForestClassifier(n_estimators=estimator, max_depth=40, n_jobs=-1)

   rf.fit(X_tf_train, y_train)

   train_pred = rf.predict(X_tf_train)

   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)

   roc_auc = auc(false_positive_rate, true_positive_rate)

   train_results.append(roc_auc)

   y_pred = rf.predict(X_tf_test)

   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)

   roc_auc = auc(false_positive_rate, true_positive_rate)

   test_results.append(roc_auc)





from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(n_estimators, train_results, 'b', label='Train AUC')

line2, = plt.plot(n_estimators, test_results, 'r', label='Test AUC')

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('AUC score')

plt.xlabel('n_estimators')

plt.show()
#Load all the libararies

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Dense, Embedding, LSTM

from keras.utils.np_utils import to_categorical

from keras.callbacks import ModelCheckpoint

from keras.models import load_model

from keras.optimizers import Adam
# Use the Keras tokenizer

num_words = 2000

tokenizer = Tokenizer(num_words=num_words)

tokenizer.fit_on_texts(X)
# Pad the data so they have the same lengths

# 

X_padded = tokenizer.texts_to_sequences(X.values)

X_padded = pad_sequences(X_padded, maxlen=800)
# Build out our simple LSTM

embed_dim = 128

lstm_out = 196



# Model saving callback

ckpt_callback = ModelCheckpoint('keras_model', 

                                 monitor='val_loss', 

                                 verbose=1, 

                                 save_best_only=True, 

                                 mode='auto')



model = Sequential()

model.add(Embedding(num_words, embed_dim, input_length = X_padded.shape[1]))

model.add(LSTM(lstm_out, recurrent_dropout=0.2, dropout=0.2))

model.add(Dense(2,activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['binary_crossentropy'])

print(model.summary())
Y = pd.get_dummies(y).values

X_train_keras, X_test_keras, Y_train_keras, Y_test_keras = train_test_split(X_padded, Y, test_size = 0.2, random_state = random_seed, stratify=Y)

print(X_train_keras.shape, Y_train_keras.shape)

print(X_test_keras.shape, Y_test_keras.shape)
batch_size = 32

model.fit(X_train_keras, Y_train_keras, epochs=8, batch_size=batch_size, validation_split=0.2, callbacks=[ckpt_callback])
model = load_model('keras_model')

keras_preds = model.predict(X_test_keras)

mat_keras = confusion_matrix(Y_test_keras.argmax(axis = 1), keras_preds.argmax(axis = 1))

sns.heatmap(mat_keras.T, square=True, annot=True, fmt='d', cbar=False, xticklabels = tick_labels, yticklabels=tick_labels)