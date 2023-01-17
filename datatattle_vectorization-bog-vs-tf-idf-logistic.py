import pandas as pd
import numpy as np
import re 
import nltk 
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
train=pd.read_csv("/kaggle/input/covid-19-nlp-text-classification/Corona_NLP_train.csv",encoding='latin1')
test=pd.read_csv("/kaggle/input/covid-19-nlp-text-classification/Corona_NLP_test.csv",encoding='latin1')
train['text'] = train.OriginalTweet
train["text"] = train["text"].astype(str)

test['text'] = test.OriginalTweet
test["text"] = test["text"].astype(str)

# Data has 5 classes, let's convert them to 3

def classes_def(x):
    if x ==  "Extremely Positive":
        return "2"
    elif x == "Extremely Negative":
        return "0"
    elif x == "Negative":
        return "0"
    elif x ==  "Positive":
        return "2"
    else:
        return "1"
    

train['label']=train['Sentiment'].apply(lambda x:classes_def(x))
test['label']=test['Sentiment'].apply(lambda x:classes_def(x))


train.label.value_counts(normalize= True)
x_train = train.text
y_train = train.label
#Using NLTK
text = "what is going to happen next in data science \
is a mystery what has happened is history it is an \
interdisciplinary field that uses scientific method \
processes algorithms and systems to extract knowledge \
and insights from many structural and unstructured data \
data science is related to data mining machine learning and big data"
txt = nltk.sent_tokenize(text)

word2count = {} 
for data in txt: 
    words = nltk.word_tokenize(data) 
    for word in words: 
        if word not in word2count.keys(): 
            word2count[word] = 1
        else: 
            word2count[word] += 1

print(word2count)
import heapq 
freq_words = heapq.nlargest(200, word2count, key=word2count.get)
freq_words
X = [] 
for data in txt: 
    vector = [] 
    for word in freq_words: 
        if word in nltk.word_tokenize(data): 
            vector.append(1) 
        else: 
            vector.append(0) 
    X.append(vector) 
X = np.asarray(X)
X
#Using SkLearn
text = "Natural Language Processing (NLP) is a sub-field of artificial intelligence \
that deals understanding and processing human language. In light of new advancements \
in machine learning, many organizations have begun applying natural language processing \
for translation, chatbots and candidate filtering"

count_vec = CountVectorizer()
count_occurs = count_vec.fit_transform([text])
count_occur_df = pd.DataFrame((count, word) for word, count in zip(count_occurs.toarray().tolist()[0], count_vec.get_feature_names()))
count_occur_df.columns = ['Word', 'Count']
count_occur_df.sort_values('Count', ascending=False, inplace=True)
count_occur_df.head()
text = "Natural Language Processing (NLP) is a sub-field of artificial intelligence \
that deals understanding and processing human language. In light of new advancements \
in machine learning, many organizations have begun applying natural language processing \
for translation, chatbots and candidate filtering"

norm_count_vec = TfidfVectorizer(use_idf=False, norm='l2')
norm_count_occurs = norm_count_vec.fit_transform([text])
norm_count_occur_df = pd.DataFrame((count, word) for word, count in zip(
    norm_count_occurs.toarray().tolist()[0], norm_count_vec.get_feature_names()))
norm_count_occur_df.columns = ['Word', 'Count']
norm_count_occur_df.sort_values('Count', ascending=False, inplace=True)
norm_count_occur_df.head()
text = "Natural Language Processing (NLP) is a sub-field of artificial intelligence \
that deals understanding and processing human language. In light of new advancements \
in machine learning, many organizations have begun applying natural language processing \
for translation, chatbots and candidate filtering"

tfidf_vec = TfidfVectorizer()
tfidf_count_occurs = tfidf_vec.fit_transform([text])
tfidf_count_occur_df = pd.DataFrame((count, word) for word, count in zip(
    tfidf_count_occurs.toarray().tolist()[0], tfidf_vec.get_feature_names()))
tfidf_count_occur_df.columns = ['Word', 'Count']
tfidf_count_occur_df.sort_values('Count', ascending=False, inplace=True)
tfidf_count_occur_df.head()
stop_words = ['a', 'an', 'the']

# Basic cleansing
def cleansing(text):
    # Tokenize
    tokens = text.split(' ')
    # Lower case
    tokens = [w.lower() for w in tokens]
    # Remove stop words
    tokens = [w for w in tokens if w not in stop_words]
    return ' '.join(tokens)

# All-in-one preproce
def preprocess_x(x):
    processed_x = [cleansing(text) for text in x]
    
    return processed_x

def build_model(mode):
    # Intent to use default paramaters for show case
    vect = None
    if mode == 'count':
        vect = CountVectorizer()
    elif mode == 'tf':
        vect = TfidfVectorizer(use_idf=False, norm='l2')
    elif mode == 'tfidf':
        vect = TfidfVectorizer()
    else:
        raise ValueError('Mode should be either count or tfidf')
    
    return Pipeline([
        ('vect', vect),
        ('clf' , LogisticRegression(solver='newton-cg',n_jobs=-1))
    ])

def pipeline(x, y, mode):
    processed_x = preprocess_x(x)
    
    model_pipeline = build_model(mode)
    cv = KFold(n_splits=5, shuffle=True)
    
    scores = cross_val_score(model_pipeline, processed_x, y, cv=cv, scoring='accuracy')
    print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
    
    return model_pipeline
x = preprocess_x(x_train)
y = y_train
    
model_pipeline = build_model(mode='count')
model_pipeline.fit(x, y)

print('Number of Vocabulary: %d'% (len(model_pipeline.named_steps['vect'].get_feature_names())))
print('Using Count Vectorizer------')
model_pipeline = pipeline(x_train, y_train, mode='count')

print('Using TF Vectorizer------')
model_pipeline = pipeline(x_train, y_train, mode='tf')

print('Using TF-IDF Vectorizer------')
model_pipeline = pipeline(x_train, y_train, mode='tfidf')