import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
#from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud,STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from bs4 import BeautifulSoup
import re,string,unicodedata
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import LancasterStemmer,WordNetLemmatizer
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from string import punctuation
from nltk import pos_tag
from nltk.corpus import wordnet
import keras
from keras.layers import Dense,LSTM
from keras.models import Sequential
df = pd.read_csv("../input/amazon-music-reviews/Musical_instruments_reviews.csv")
df.head()
df.columns
df.isna().sum()
df.reviewText.fillna("",inplace = True)
del df['reviewerID']
del df['asin']
del df['reviewerName']
del df['helpful']
del df['unixReviewTime']
del df['reviewTime']
df.head()
df["quality"] = df.loc[:,"overall"].apply(lambda x : "good" if x >= 4 else ("neutral" if x==3 else "bad" ))
df["strQuality"] = df.loc[:,"quality"].apply(lambda x : 2 if x == "good" else (1 if x== "neutral" else 0 ))
df.head()
df['text'] = df['reviewText'] + ' ' + df['summary']
del df['reviewText']
del df['summary']
df.overall.value_counts()
for i,each in enumerate(df.overall.value_counts()):
    print(f"Percentage of {df.overall.value_counts().index[i]} stars : {(each*100/len(df.overall)):.2f}")
stop = set(stopwords.words('english'))
punctuation = list(string.punctuation)
stop.update(punctuation) ### adding the punctioation as stopwords as well!
stop
rem = ["aren't", "aren", "but", "couldn", "couldn't", "don", "don't","didn", "didn't", "doesn", "doesn't", "wouldn", "wouldn't", "won", "won't", "weren", "weren't", "wasn", "wasn't", "should", "shouldn't", "needn", "needn't", "mustn", "mustn't", "mightn", "mightn't", "isn", "isn't", "haven", "haven't", "hasn", "hasn't", "hadn", "hadn't","not", "no"]
def get_simple_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
lemmatizer = WordNetLemmatizer()
def lemmatize_words(text):
    final_text = []
    for i in text.split():
        #print("word : ", i)
        if i.strip().lower() not in stop:
            pos = pos_tag([i.strip()])
            #print("pos : ", pos)
            word = lemmatizer.lemmatize(i.strip(),get_simple_pos(pos[0][1]))
            #print("Lemma word : ", word)
            final_text.append(word.lower())
    return " ".join(final_text)
z = ["I don't knowing know know, but don't care","I would like you know","Don't care care cares"]
for each in z:
    r = lemmatize_words(each)
    print("our r : ", r)
# for i in range(100):
#     if df.loc[i,"quality"] == "good":
#         print(i,"\n", df.loc[i,"text"])
# for i in range(100):
#     if df.loc[i,"quality"] == "neutral":
#         print(i,"\n", df.loc[i,"text"])
df.text = df.text.apply(lemmatize_words)
df.head()
good = df.text[df.quality == "good"]
neutral = df.text[df.quality == "neutral"] #.drop(columns = "overall")
bad = df.text[df.quality == "bad"] # .drop(columns = "overall")
good.shape,bad.shape,neutral.shape

fig = plt.figure(figsize=(20,30))
qual = {0 : ["neutral",neutral], 1 : ["bad", bad], 2 : ["good",good]}
qual[0][0]
for i in range(3):
    ax = fig.add_subplot(1,3,i+1)
    wc = WordCloud(min_font_size = 3,  max_words = 3000 , width = 1600 , height = 800).generate(" ".join(qual[i][1]))
    #wc.recolor(color_func = grey_color_func)
    ax.imshow(wc,interpolation = 'bilinear')
    plt.xlabel(qual[i][0])
    #ax.axis('off')
from sklearn.feature_extraction.text import CountVectorizer

from yellowbrick.text import FreqDistVisualizer
from yellowbrick.datasets import load_hobbies

for i in range(3):  
    fig = plt.figure(figsize=(15,3))
    corpus = qual[i][1]
    vectorizer = CountVectorizer()
    docs       = vectorizer.fit_transform(corpus)
    features   = vectorizer.get_feature_names()

    visualizer = FreqDistVisualizer(features=features, orient='v',n=10, title=["Frequency of 10 words for : " + qual[i][0]])
    visualizer.fit(docs)    
    visualizer.show()
for i in range(3):  
    fig = plt.figure(figsize=(15,3))
    corpus = qual[i][1]
    vectorizer = CountVectorizer(min_df=0,binary=False,ngram_range=(2,3)) ### We changed this parameter!!
    docs       = vectorizer.fit_transform(corpus)
    features   = vectorizer.get_feature_names()

    visualizer = FreqDistVisualizer(features=features, orient='v',n=10, title=["Frequency of 10 words for : " + qual[i][0]])
    visualizer.fit(docs)    
    visualizer.show()
#### Remember our "z" list:
#### z = ["I don't knowing know know, but don't care","I would like you know","Don't care care cares"]
cvz=CountVectorizer(min_df=0,max_df=1,binary=False,ngram_range=(2,3))
#cv=CountVectorizer(ngram_range=(2,3))

cv_testz=cvz.fit_transform(z)
cvz.get_feature_names() ### Take a look at what this method does:
cv_testz.toarray()
cvz.vocabulary_["you know"]
a = cv_testz.toarray()
for i in range(3):
       a[i][18] = a[i][18]*2 
display(cv_testz.toarray())
display(a)
#z = ["I don't know know know, but don't care","I like you know","Don't care care care"]
tvt=TfidfVectorizer(min_df=0,max_df=1,use_idf=True,ngram_range=(2,3))
#transformed reviews
tvt_test=tvt.fit_transform(z)

tvt.get_feature_names()
tvt_test.toarray()
#z = ["I don't know know know, but don't care","I like you know","Don't care care care"]

x_train,x_test,y_train,y_test = train_test_split(df.text,df.quality,test_size = 0.2 , random_state = 0)
cv=CountVectorizer(min_df=0,binary=False,ngram_range=(2,3))
#cv=CountVectorizer(ngram_range=(2,3))
#transformed train reviews
cv_train_reviews=cv.fit_transform(x_train)
#transformed reviews
cv_test_reviews=cv.transform(x_test)
#display(cv_train_reviews.toarray())
print('cv_train:',cv_train_reviews.shape)
print('cv_test:',cv_test_reviews.shape)

tv=TfidfVectorizer(min_df=0,use_idf=True,ngram_range=(2,3))
#transformed reviews
tv_train_reviews=tv.fit_transform(x_train)

tv_test_reviews=tv.transform(x_test)
print('Tfidf_train:',tv_train_reviews.shape)
print('Tfidf_test:',tv_test_reviews.shape)

lr=LogisticRegression(penalty='l2',max_iter=500,C=1,random_state=0)
#Fitting
lr_bow=lr.fit(cv_train_reviews,y_train)
print(lr_bow)

lr_tfidf=lr.fit(tv_train_reviews,y_train)
print(lr_tfidf)

#Predicting 
lr_bow_predict=lr.predict(cv_test_reviews)

lr_tfidf_predict=lr.predict(tv_test_reviews)

#Accuracy score 
lr_bow_score=accuracy_score(y_test,lr_bow_predict)
print("lr_bow_score :",lr_bow_score)

lr_tfidf_score=accuracy_score(y_test,lr_tfidf_predict)
print("lr_tfidf_score :",lr_tfidf_score)
#report
lr_bow_report=classification_report(y_test,lr_bow_predict,target_names=['good','neutral','bad'])
print(lr_bow_report)


lr_tfidf_report=classification_report(y_test,lr_tfidf_predict,target_names=['good','neutral','bad'])
print(lr_tfidf_report)
lr_bow_report=classification_report(y_test,lr_bow_predict,target_names=['good','neutral','bad'])
print(lr_bow_report)


lr_tfidf_report=classification_report(y_test,lr_tfidf_predict,target_names=['good','neutral','bad'])
print(lr_tfidf_report)
cv.vocabulary_["sound good"]
def tuning(pen, inc):
    global tr, te
    tr = cv_train_reviews.toarray()
    te = cv_test_reviews.toarray()

    voc = ["planet waves", "sound like", "work well", "sound good", "they re"] ## work well we're goind to penalize

    for each in voc:
        idx = cv.vocabulary_[each]
        if each == "work well": #### PENALIZING
            for i in range(cv_train_reviews.shape[0]):
                tr[i][idx] = int(tr[i][idx]//pen) 
            for i in range(cv_test_reviews.shape[0]):
                te[i][idx] = int(te[i][idx]//pen)
        else:##### INCREASING THE WEIGHT
            for i in range(cv_train_reviews.shape[0]):
                tr[i][idx] = tr[i][idx]*inc 
            for i in range(cv_test_reviews.shape[0]):
                te[i][idx] = te[i][idx]*inc
    tr_sm = sparse.csr_matrix(tr)
    te_sm = sparse.csr_matrix(te)
    
    return tr_sm, te_sm
display(tr_sm)
display(cv_train_reviews)
results = {}

for inc in range(2,23,10):
    for pen in range(2,3):
        tr_sm, te_sm = tuning(pen,inc)
        Lr=LogisticRegression(penalty='l2',max_iter=500,C=1,random_state=0) ## We are already penalizing!

        lr_bow=Lr.fit(tr_sm,y_train)
        print(lr_bow)

        lr_bow_predict=Lr.predict(te_sm)

        #Accuracy score 
        lr_bow_score=accuracy_score(y_test,lr_bow_predict)
        print("lr_bow_score :",round(lr_bow_score,6))
        mod = "Model: increase " + str(inc) + ", penalize in " + str(pen)
        results[mod] = round(lr_bow_score,6)
results
df = pd.read_csv("../input/amazon-music-reviews/Musical_instruments_reviews.csv")
df.reviewText.fillna("",inplace = True)
del df['reviewerID']
del df['asin']
del df['reviewerName']
del df['helpful']
del df['unixReviewTime']
del df['reviewTime']

df["quality"] = df.loc[:,"overall"].apply(lambda x : "good" if x >= 4 else ("neutral" if x==3 else "bad" ))
df["strQuality"] = df.loc[:,"quality"].apply(lambda x : 2 if x == "good" else (1 if x== "neutral" else 0 ))

df['text'] = df['reviewText'] + ' ' + df['summary']
del df['reviewText']
del df['summary']
df.head()
for each in rem:
    stop.discard(each)
stop ## check the new list to see if it's smaller:
df.text = df.text.apply(lemmatize_words)
def final(X_data_full):
    
    cv = CountVectorizer(min_df = 0, max_features=1000, ngram_range =(2,3))
    X_full_vector = cv.fit_transform(X_data_full).toarray()    
    
    full = X_full_vector
    print("our full: ", full)
    voc = ["planet waves", "sound like", "work well", "sound good", "they re"] ## work well we're goind to penalize
    
    try:
        for each in voc:
            idx = cv.vocabulary_[each]
            if each == "work well": #### PENALIZING
                for i in range(X_full_vector.shape[0]):
                    full[i][idx] = int(full[i][idx]//2) 
            else:##### INCREASING THE WEIGHT
                for i in range(X_full_vector.shape[0]):
                    full[i][idx] = full[i][idx]*inc
    except:
        print("didn't work!")
    full_sm = sparse.csr_matrix(full)
    
    tfidf = TfidfTransformer()
    X_data_full_tfidf = tfidf.fit_transform(full_sm).toarray()
    
    return X_data_full_tfidf
    
x = final(df.text)
x_train,x_test,y_train,y_test = train_test_split(x,df.strQuality,test_size = 0.2 , random_state = 0)
XX = x_train
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional

embedding_size=32
max_words=5000

model = Sequential()
model.add(Embedding(max_words, embedding_size, input_length=1000 )) #x_train.shape[0]))
model.add(Bidirectional(LSTM(16, return_sequences = True)))
model.add(Bidirectional(LSTM(16)))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3,activation='softmax'))

print(model.summary())
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

y_train_dummies = pd.get_dummies(y_train).values
print('shape label tensor: ', y_train_dummies.shape)

#trainingggg the model
model.fit(XX, y_train, epochs=2, batch_size=32)

# display(XX.shape)
# display(XX[:int(len(XX)/5),:].shape)
# converting categorical var in y_train to numerical var
y_test_dummies = pd.get_dummies(y_test).values
print('Shape of Label tensor: ', y_test_dummies.shape)

#model = load_model('../output/MusicalInstrumentReviews_correct.h5')
scores = model.evaluate(XX[:int(len(XX)/4)+1,:], y_test)

LSTM_accuracy = scores[1]*100

print('Test accuracy: ', scores[1]*100, '%')