import pandas as pd 

import numpy as np 

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split

from  sklearn.utils import resample

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

import xgboost as xgb

import matplotlib.pyplot as plt

from nltk.sentiment import SentimentIntensityAnalyzer

from sklearn.metrics import accuracy_score, classification_report

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer

from sklearn.metrics import accuracy_score, precision_score, recall_score
#hyper parameter

sent_length = 30

embedded_vector_features = 40

dropout = 0.2

Epochs = 5

Batch_size = 64

n_neurons_1 = 64

n_neurons_2 = 32

n_neurons = 16
df = pd.read_csv("../input/amazon-reviews/train_data.csv") # reading the data 
df.head() # view dataframe
df.describe() # dataframe stats
import nltk 

import re

from nltk.corpus import stopwords
nltk.download('stopwords')
X = [i.lower().split() for i in df["reviews.text"]] # cleaning the data and creating a corpus

X[1]

from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

corpus = []

for review in X:

    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]

    corpus.append(review)
corpus[:10]
df["reviews.text"][df["sentiment"]=="Positive"].any() # Example of a positive Review 
df["reviews.text"][df["sentiment"]=="Negative"].any() # example of a negative review
df["reviews.text"][df["sentiment"]=="Neutral"].any() # example of a neutral review
# using tfidf vectorizer 

tfidf = TfidfVectorizer()

X = tfidf.fit_transform([str(word)for word in corpus])
X.get_shape()
print(X) # tfidf score 
X = X.toarray() # converting to array

y = df.sentiment.values



train_X,test_X,train_y,test_y = train_test_split(X,y,test_size = 0.2) # split into train and test



# Naive bayes model 

model = MultinomialNB()

model.fit(train_X,train_y)
print("Navie Bayes model accuracy:",model.score(test_X,test_y)) # accuracy score for naive bayes
print("Navie Bayes model prediction values:",pd.Series(model.predict(test_X)).value_counts()) # predictions value counts 
print("Navie Bayes model prediction values:",pd.Series(test_y).value_counts()) # actual value counts 
print("Navie bayes classification report:",classification_report(model.predict(test_X),test_y))
X = pd.DataFrame(X)

y = df.sentiment



train_X,test_X,train_y,test_y = train_test_split(X,y,test_size = 0.2)



X = pd.concat([train_X,train_y], axis=1)



# separate minority and majority classes

positive = X[X["sentiment"]=="Positive"]

neutral = X[X["sentiment"]=="Neutral"]

negative = X[X["sentiment"]=="Negative"]



# upsample minority

Neutral_upsampled =  resample(neutral,

                     replace=True, # sample with replacement

                     n_samples=len(positive), # match number in majority class

                     random_state=2)



Negative_upsampled = resample(negative,

                     replace=True, # sample with replacement

                     n_samples=len(positive), # match number in majority class

                     random_state=2)





upsampled = pd.concat([positive,Negative_upsampled,Neutral_upsampled])

print("Total values after upsampling:",upsampled.sentiment.value_counts()) # upsampled data 
train_X = upsampled.drop(["sentiment"],1).values # dividing back into X and y

train_y = upsampled.sentiment.values
#Naive bayes model 

    

model = MultinomialNB()

model.fit(train_X,train_y)

print("Navie Bayes model after upsampling accuracy:",model.score(test_X,test_y)) # accuracy score for naive bayes
print("Navie Bayes model prediction values after upsampling:",pd.Series(model.predict(test_X)).value_counts()) # predictions value counts 
print("Navie Bayes model prediction values after upsampling:",pd.Series(test_y).value_counts()) # actual value counts 
print("Navie bayes classification report after upsampling :",classification_report(model.predict(test_X),test_y))
print("This model has better score after using upsampling")
# Random forest Classifier

model = RandomForestClassifier()

model.fit(train_X,train_y)
print("random forest accuracy:",model.score(test_X,test_y)) # random forest model on upsampled data
print("random forest prediction values")

pd.Series(model.predict(test_X)).value_counts() # predictions value counts for random forest
print("random forest actual values")

pd.Series(test_y).value_counts() # actual value counts for random forest
print("random forest classifaction report")

print(classification_report(model.predict(test_X),test_y))
print("""The random forest model has better F1 scores for Negative and Positive

      this model may be more useful since better classification of Better 

      Negative and Positive reviews will provide

      more insights.""")
# svm model

model = SVC()

model.fit(train_X,train_y)
print("SVC accuracy:",model.score(test_X,test_y)) # SVC model on upsampled data
print("SVM prediction values")

pd.Series(model.predict(test_X)).value_counts() # predictions value counts for SUPPORT VECTOR MACHINE 
print("svm actual values")

pd.Series(test_y).value_counts() # actual value counts for svm
print("SVM classifaction report")

print(classification_report(model.predict(test_X),test_y)) 
train_y = pd.Series(train_y) # converting into integers for xgb

train_y.replace("Positive",1,inplace=True)

train_y.replace("Neutral",0,inplace=True)

train_y.replace("Negative",-1,inplace=True)



test_y.replace("Positive",1,inplace=True)

test_y.replace("Neutral",0,inplace=True)

test_y.replace("Negative",-1,inplace=True)
model=xgb.XGBClassifier(random_state=1,learning_rate=0.1)

model.fit(train_X, train_y)

print("XGB accuracy:",model.score(test_X.values,test_y))# XGB model on upsampled data
print("XGB prediction values")

pd.Series(model.predict(test_X.values)).value_counts() # predictions value counts for XGB
print("XGB actual values")

pd.Series(test_y).value_counts() # actual value counts for XGB
print("XGB classifaction report")

print(classification_report(model.predict(test_X.values),test_y)) 
sent = SentimentIntensityAnalyzer() # using the sent analyser to add more features

sentiment_score = df["reviews.text"].apply(sent.polarity_scores)



# using tfidf vectorizer 

tfidf = TfidfVectorizer()

X = tfidf.fit_transform(df["reviews.text"].values)
sentiment_score
X = X.toarray()

X = pd.DataFrame(X)



X["postive_score"] = [i["pos"] for i in sentiment_score]

X["negative_score"] = [i["neg"] for i in sentiment_score]  # feature engineering

X["neutral_score"] = [i["neu"] for i in sentiment_score]



y = df.sentiment



train_X,test_X,train_y,test_y = train_test_split(X,y,test_size = 0.2)



X = pd.concat([train_X, train_y], axis=1)



# separate minority and majority classes

positive = X[X["sentiment"]=="Positive"]

neutral = X[X["sentiment"]=="Neutral"]

negative = X[X["sentiment"]=="Negative"]



# upsample minority

Neutral_upsampled =  resample(neutral,

                     replace=True, # sample with replacement

                     n_samples=len(positive), # match number in majority class

                     random_state=2)



Negative_upsampled = resample(negative,

                     replace=True, # sample with replacement

                     n_samples=len(positive), # match number in majority class

                     random_state=2)





upsampled = pd.concat([positive,Negative_upsampled,Neutral_upsampled])

upsampled.head()
train_X = upsampled.drop(["sentiment"],axis=1).values # diving into X and y again 

train_y = upsampled.sentiment.values



model=xgb.XGBClassifier(random_state=1,learning_rate=0.1) # training the model with new features 

model.fit(train_X, train_y)
pd.Series(model.predict(test_X.values)).value_counts()
pd.Series(test_y).value_counts()
print("New XGB classifaction report")

print(classification_report(model.predict(test_X.values),test_y)) 
# neural network model. 

from tensorflow.keras.layers import Dense, LSTM, Embedding, Reshape, Flatten, Dropout, GRU

from tensorflow.keras.models import Sequential 

from tensorflow import keras

from tensorflow.keras.preprocessing.text import one_hot

from tensorflow.keras.preprocessing.sequence import pad_sequences


X = df["reviews.text"].values 

y  = df["sentiment"]
vocab_size = len(corpus)
one_hot_repr = [one_hot(str(word),vocab_size) for word in corpus] 

one_hot_repr[:10]


embedded_doc = pad_sequences(one_hot_repr,padding="pre",maxlen = sent_length)
embedded_doc
X = np.array(embedded_doc)

y = df["sentiment"].values

train_X,test_X,train_y,test_y = train_test_split(X,y,test_size = 0.2) # split into train and test

train_y = pd.Series(train_y) # replacing the values with integers to prep for keras lstm 

train_y.replace("Positive",0,inplace=True)

train_y.replace("Neutral",1,inplace=True)

train_y.replace("Negative",2,inplace=True)



test_y = pd.Series(test_y)



test_y.replace("Positive",0,inplace=True)

test_y.replace("Neutral",1,inplace=True)

test_y.replace("Negative",2,inplace=True)


model = Sequential()

model.add(Embedding(len(corpus),embedded_vector_features,input_length= sent_length))

model.add(LSTM(n_neurons_1, return_sequences=True,activation="relu"))

model.add(LSTM(n_neurons_2,activation="relu"))

model.add(Dropout(dropout))



model.add(Dense(3,activation="relu"))

model.compile(optimizer = "adam",loss = "sparse_categorical_crossentropy",

              metrics = ["accuracy"])
model.fit(train_X,train_y,batch_size=Batch_size,epochs = Epochs)
print("LSTM model (without upsampling) classifaction report")

print(classification_report(model.predict_classes(test_X),test_y)) 
print("""The model with keras is only predicting 0's as seen in the classification report

Next trying upsampling to fix this problem """)
# upsampling data 

X = pd.DataFrame(X)



y = df.sentiment



train_X,test_X,train_y,test_y = train_test_split(X,y,test_size = 0.2)



X = pd.concat([train_X,train_y], axis=1)



# separate minority and majority classes

positive = X[X["sentiment"]=="Positive"]

neutral = X[X["sentiment"]=="Neutral"]

negative = X[X["sentiment"]=="Negative"]



# upsample minority

Neutral_upsampled =  resample(neutral,

                     replace=True, # sample with replacement

                     n_samples= len(positive), # match number in majority class

                     random_state=2)



Negative_upsampled = resample(negative,

                     replace=True, # sample with replacement

                     n_samples= len(positive), # match number in majority class

                     random_state=2)





upsampled = pd.concat([positive,Negative_upsampled,Neutral_upsampled])

train_X = upsampled.drop(["sentiment"],axis=1)

train_y = upsampled["sentiment"]
train_y = pd.Series(train_y) # changing into integers

train_y.replace("Positive",0,inplace=True)

train_y.replace("Neutral",1,inplace=True)

train_y.replace("Negative",2,inplace=True)



test_y = pd.Series(test_y)



test_y.replace("Positive",0,inplace=True)

test_y.replace("Neutral",1,inplace=True)

test_y.replace("Negative",2,inplace=True)
# neural network model - LSTM

model = Sequential()

model.add(Embedding(len(corpus),embedded_vector_features,input_length= sent_length))

model.add(LSTM(n_neurons_1, return_sequences=True,activation="relu"))

model.add(LSTM(n_neurons_2,activation="relu"))

model.add(Dense(n_neurons,activation="relu"))

model.add(Dense(128,activation= "relu"))

#model.add(Dropout(0.4))

model.add(Dense(3,activation="relu"))

model.compile(optimizer = "adam",loss = "sparse_categorical_crossentropy",

              metrics = ["accuracy"])
model.fit(train_X,train_y,batch_size=Batch_size,epochs = Epochs)
accuracy_score(model.predict_classes(train_X),train_y)
print("Keras LSTM classifaction report")

print(classification_report(model.predict_classes(test_X),test_y)) 
# neural network model - GRU

model = Sequential()

model.add(Embedding(len(corpus),embedded_vector_features,input_length= sent_length))

model.add(GRU(64, return_sequences=True,activation="relu"))

model.add(GRU(64,activation="sigmoid"))





model.add(Dense(3,activation="relu"))

model.compile(optimizer = "adam",loss = "sparse_categorical_crossentropy",

              metrics = ["accuracy"])
model.fit(train_X,train_y,batch_size=Batch_size,epochs = Epochs)
print("Keras GRU classifaction report")

print(classification_report(model.predict_classes(test_X),test_y)) 
sent_length = 30

embedded_vector_features = 40

dropout = 0.2

Epochs = 5

Batch_size = 64

n_neurons_1 = 64

n_neurons_2 = 32

n_neurons = 16
def create_model(neurons,neurons2):

    # create model

    model = Sequential()

    model.add(Embedding(len(corpus),embedded_vector_features,input_length= sent_length))

    model.add(LSTM(neurons, return_sequences=True,activation="relu"))

    model.add(LSTM(neurons2,activation="relu"))

    model.add(Dense(32,activation="relu"))



    model.add(Dense(3,activation="relu"))

    model.compile(optimizer = "adam",loss = "sparse_categorical_crossentropy",

                  metrics = ["accuracy"])





    return model
scorers = {

        'precision_score': make_scorer(precision_score),

        'recall_score': make_scorer(recall_score),

        'accuracy_score': make_scorer(accuracy_score)

        }
model = KerasClassifier(build_fn=create_model,epochs=3,batch_size=64,verbose=0)


n_neurons = [5,10,]

n_neurons_2 = [16,32,]



parameters =  {"neurons": n_neurons,"neurons2":n_neurons_2}

grid = GridSearchCV(model,parameters)

grid_result = grid.fit(train_X, train_y)

grid_result
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))
pd.Series(grid.predict(train_X)).value_counts()