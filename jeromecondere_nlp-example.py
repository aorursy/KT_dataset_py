import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline



np.random.seed(1337)



from keras.models import Sequential

from keras.layers import Embedding, Dense, LSTM, BatchNormalization

from keras.layers import SpatialDropout1D

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

chatbot = pd.read_csv("../input/deepnlp/Sheet_1.csv",usecols=['response_id','class','response_text'],encoding='latin-1')

resume = pd.read_csv("../input/deepnlp/Sheet_2.csv",encoding='latin-1')
chatbot.head(5)
resume.head(5)
count_vect = CountVectorizer()





x = chatbot['response_text']

y = chatbot['class']

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=1)

X_train_counts = count_vect.fit_transform(x_train)

X_test_counts = count_vect.transform(x_test)

tfidf_transformer = TfidfTransformer()

X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

X_test_tfidf = tfidf_transformer.fit_transform(X_test_counts)



print(X_train_tfidf.shape)

print(X_test_tfidf.shape)
# from sklearn.feature_extraction.text import TfidfTransformer

# tfidf_transformer = TfidfTransformer()

# X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# X_train_tfidf.shape
from sklearn.naive_bayes import MultinomialNB



naive = MultinomialNB().fit(X_train_tfidf, y_train)

predicted = naive.predict(X_test_tfidf)

np.mean(predicted == y_test)
from sklearn.linear_model import SGDClassifier



svm = Pipeline([('vect', CountVectorizer()),

	('tfidf', TfidfTransformer()),

    ('clf-svm', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter_no_change=5, random_state=42))

])

svm_fit = svm.fit(x_train, y_train)

svm_predict = svm.predict(x_test)

np.mean(svm_predict == y_test)
print(resume.shape)



x_resume = resume['resume_text']

y_resume = resume['class']
# The maximum number of words to be used. (most frequent)

MAX_NB_WORDS = 50000

# Max number of words in each complaint.

MAX_SEQUENCE_LENGTH = 300

# This is fixed.

EMBEDDING_DIM = 100



tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)

tokenizer.fit_on_texts(x_resume)

word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))
#truncate and pad sequence to have string representation of each size

X = tokenizer.texts_to_sequences(x_resume)

X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

print('Shape of data tensor:', X.shape)



#convert label to categorical values

Y = pd.get_dummies(y_resume).values

print('Shape of label tensor:', Y.shape)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, random_state = 42)

print(X_train.shape,Y_train.shape)

print(X_test.shape,Y_test.shape)
model = Sequential()

model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))

model.add(SpatialDropout1D(0.2))

model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



epochs = 8



history = model.fit(X_train, Y_train, epochs=epochs,validation_split=0.1)
accr = model.evaluate(X_test,Y_test)

print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
plt.title('Loss')

plt.plot(history.history['loss'], label='train')

plt.plot(history.history['val_loss'], label='test')

plt.legend()

plt.show();