# @author: Himanshu Choudhary

# @homepage : http://www.himanshuchoudhary.com/

# @git : https://bitbucket.org/himanshuchoudhary/



%matplotlib inline

import numpy as np

import pandas as pd

import matplotlib



from gensim import summarization

from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.cross_validation import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from math import floor,ceil



from sklearn.svm import LinearSVC



from keras.models import Sequential

from keras.layers import LSTM, Dense, Embedding
def rating_to_polarity(rating):

    if rating > 3:

        return 1

    return 0



def score_to_polarity(score):

    if score < 0:

        return 0

    return 1



def score_to_rating(score):

    rating = score*2+3

    return int(round(rating))



def get_keywords(text):

    try:

        keywords = summarization.keywords(text,ratio=1.0,split=True)

    except Exception:

        keywords = []

    return ' '.join(keywords)



def categorize(ratings):

    cats = []

    for rating in ratings:

        v = [0,0,0,0,0]

        v[rating-1] = 1

        cats.append(v)

    return np.array(cats)



def generate_random_rating():

    a = np.random.randint(low=1,high=6,size=1)

    return np.mean(a,dtype=np.int32)
data = pd.read_csv('../input/Reviews.csv',header=0,index_col=0,encoding='utf-8')

data = data.sample(n=10000,random_state=1)

# data = data[data.Score != 3]    # for polarity classification

data = data.dropna(how='any')
# reviews = data.Summary

reviews = data.Text

# reviews = data.Text.map(get_keywords)

ratings = data.Score
vectorizer = TfidfVectorizer(max_df=.8)

vectorizer.fit(reviews)
X = vectorizer.transform(reviews).toarray()

# y = ((ratings-3)/2.0).values    # for polarity score

# y = ratings.map(rating_to_polarity).values      # for polarity classification

y = categorize(ratings.values)   # for rating classification



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2)
model = Sequential()

model.add(Dense(128,input_dim=X_train.shape[1]))



# for polarity classification

# model.add(Dense(1,activation='sigmoid'))

# model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])



# for rating classification

model.add(Dense(5,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])



# for polarity score

# model.add(Dense(1,activation='tanh'))

# model.compile(loss='mean_squared_error',optimizer='rmsprop',metrics=['mean_squared_error'])



model.fit(X_train,y_train,nb_epoch=10,batch_size=32,verbose=1)

model.evaluate(X_test,y_test)[1]
preds = model.predict(X_test)

out = []

for i in range(len(preds)):

#     out.append([score_to_rating(preds[i][0]),int(y_test[i]*2+3)])     # for polarity score

#     out.append([int(round(preds[i][0])),y_test[i]])     # for polarity classification

    out.append([preds[i].argmax()+1,y_test[i].argmax()+1])    # for rating classification



out = pd.DataFrame(out,columns=['PredictedRating','ActualRating'])

out['DifferenceActPred'] = (out.ActualRating - out.PredictedRating).map(abs)
out[['ActualRating','PredictedRating']].hist()

out[['DifferenceActPred']].hist()
print("Dataset size : {:d}".format(len(data)))

print("Training set size : {:d}".format(len(X_train)))

print("Testing set size : {:d}".format(len(X_test)))

print("Accuracy between predicted and actual : {:f}".format(accuracy_score(out.PredictedRating,out.ActualRating)))

print("Accuracy with +-1 difference between predicted and actual : {:f}".format(float(out.DifferenceActPred.value_counts()[0]+out.DifferenceActPred.value_counts()[1])/len(out)))