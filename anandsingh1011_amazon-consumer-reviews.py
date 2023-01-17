from tensorflow.keras.layers import Dense, Input

from tensorflow.keras import optimizers

from tensorflow.keras import losses

from tensorflow.keras import metrics

from tensorflow.keras import activations

from tensorflow.keras import models

from tensorflow.keras.utils import to_categorical

import tensorflow as tf

from tensorflow import keras

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('ggplot')

WIDE = (12,6)

WIDER = (16,8)

LONG = (8, 12)

LONGER = (8, 18)

plt.rcParams['figure.figsize'] = WIDER



%matplotlib inline
df = pd.read_csv('../input/consumer-reviews-of-amazon-products/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv')
df.head()
df.columns
df_new = df[[ 'name', 'brand',

       'categories', 'primaryCategories', 

        'reviews.doRecommend',

       'reviews.numHelpful', 'reviews.rating', 

       'reviews.text', 'reviews.title', 'reviews.username']]
df_new.head(3)
df_reviews = df[[ 'reviews.rating','reviews.text', 'reviews.title',]]


df_reviews.head(3)
df_reviews.shape
df_classify = df_reviews[df_reviews["reviews.rating"].notnull()]

df_classify["sentiment"] = df_classify["reviews.rating"] >= 4

df_classify["sentiment"] = df_classify["sentiment"].replace([True , False] , ["Postive" , "Negative"])



# Lets count positive and negative review

df_classify["sentiment"].value_counts().plot.bar()
# Cleaning the texts

import re

import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

corpus = []
for i in range(0, 5000):

    review = re.sub('[^a-zA-Z]', ' ', df_reviews['reviews.text'][i])

    review = review.lower()

    review = review.split()

    ps = PorterStemmer()

    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]

    review = ' '.join(review)

    corpus.append(review)
corpus=pd.DataFrame(corpus, columns=['Reviews']) 

corpus.head()
result=corpus.join(df_reviews[['reviews.rating']])

result.head()
def formatt(x):

    if x < 4:

        return 0

    if x >= 4:

        return 1

vfunc = np.vectorize(formatt)



result['sentiment'] = result['reviews.rating'].map(vfunc)
result.head(3)
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()

tfidf.fit(result['Reviews'])
from sklearn.model_selection import train_test_split

X = tfidf.transform(result['Reviews'])

y = result['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)

print(X_train.shape,y_train.shape)

print(X_test.shape,y_test.shape)
prediction =  {}
from sklearn.naive_bayes import BernoulliNB

model1 = BernoulliNB().fit(X_train , y_train)

y_pred_bernoulli = model1.predict_proba(X_test)



prediction['BernoulliNB'] = model1.predict_proba(X_test)[:,1]

print("BernoulliNB Accuracy : {}".format(model1.score(X_test , y_test)))

from sklearn.naive_bayes import MultinomialNB

model2 = MultinomialNB().fit(X_train , y_train)

y_pred_multinomial = model2.predict_proba(X_test)



prediction['Multinomial'] = model2.predict_proba(X_test)[:,1]

print("Multinomial Accuracy : {}".format(model2.score(X_test , y_test)))

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(solver='lbfgs', multi_class='auto' , max_iter=4000)

logistic = logreg.fit(X_train , y_train)

prediction['LogisticRegression'] = logreg.predict_proba(X_test)[:,1]

#y_pred_logistic = logreg.decision_function(X_test)

y_pred_logistic = logreg.predict(X_test)

print("Logistic Regression Accuracy : {}".format(logreg.score(X_test , y_test)))
from sklearn.svm import SVC

svcreg = SVC(kernel = 'rbf', random_state = 4)

svc = svcreg.fit(X_train , y_train)

prediction['SVC'] = logreg.predict_proba(X_test)[:,1]



y_pred_svm = svcreg.decision_function(X_test)



print("SVC Accuracy : {}".format(svcreg.score(X_test , y_test)))
from sklearn.metrics import roc_curve,auc

colors_counter = 0

colors_code = ['b', 'g', 'y', 'm', 'k']

for model, predicted in prediction.items():

    fpr, tpr, thresholds = roc_curve(y_test, predicted)

    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, colors_code[colors_counter], label='%s: AUC %0.2f'% (model,roc_auc))

    colors_counter += 1



plt.rcParams['figure.figsize'] = WIDER

plt.title('Classifiers Comparaison With ROC')

plt.legend(loc='lower right')

#plt.plot([0,1],[0,1],'r--')

#plt.xlim([-0.1,1.2])

#plt.ylim([-0.1,1.2])

plt.xlabel('False Positive Rate -->')

plt.ylabel('True Positive Rate -->')

plt.show()
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()

tfidf.fit(result['Reviews'])


from wordcloud import STOPWORDS



from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import CountVectorizer

stopwords = set(STOPWORDS)

stopwords.remove("not")



count_vect = CountVectorizer(min_df=2 ,stop_words=stopwords , ngram_range=(1,2))

tfidf_transformer = TfidfTransformer()



df_cv = count_vect.fit_transform(result["Reviews"])        



df_tf = tfidf_transformer.fit_transform(df_cv)
from sklearn.feature_extraction.text import CountVectorizer

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Embedding, LSTM

from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical

import re



max_fatures = 30000

tokenizer = Tokenizer(nb_words=max_fatures, split=' ')

tokenizer.fit_on_texts(result['Reviews'].values)

X1 = tokenizer.texts_to_sequences(result['Reviews'].values)

X1 = pad_sequences(X1)



Y1 = pd.get_dummies(result['reviews.rating']).values

X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1,Y1, random_state = 42)

print(X1_train.shape,Y1_train.shape)

print(X1_test.shape,Y1_test.shape)



embed_dim = 150

lstm_out = 200



model = Sequential()

model.add(Embedding(max_fatures, embed_dim,input_length = X1.shape[1]))

model.add(LSTM(lstm_out))

model.add(Dense(5,activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

print(model.summary())
batch_size = 512
model.fit(X1_train, Y1_train, epochs = 50, batch_size=batch_size, validation_split=0.3 , verbose = 2)
history = model.history
score,acc = model.evaluate(X1_test, Y1_test, verbose = 2, batch_size = batch_size)

print("score: %.2f" % (score))

print("acc: %.2f" % (acc))
print(history.history.keys())
# Plot training & validation accuracy values

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()