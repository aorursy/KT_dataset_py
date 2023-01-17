import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from spacy.lang.en import STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from collections import Counter
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
def load_data():
    train = pd.read_csv("../input/janatahack-independence-day-2020-ml-hackathon/train.csv")
    test = pd.read_csv("../input/janatahack-independence-day-2020-ml-hackathon/test.csv")
    sample = pd.read_csv("../input/janatahack-independence-day-2020-ml-hackathon/sample_submission_UVKGLZE.csv")
    train.drop("ID", axis=1, inplace=True)
    test.drop("ID", axis=1, inplace=True)
    print(f"Train data shape : {train.shape}")
    print(f"Test data shape : {test.shape}")
    return (train, test, sample)
train, test, sample = load_data()
train.head()
target = ['Computer Science','Physics','Mathematics','Statistics','Quantitative Biology','Quantitative Finance']
train[target].sum()
import re
def url(text):
    url_check = "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    clean = re.sub(url_check, "", text)
    return clean

def num_repl(text):
    num_check = "([0-9,]*)"
    clean = re.sub(num_check, "", text).strip()
    return clean

def remove_stop(text):
    word_list = text.lower().split()
    stopword_dict = Counter(STOP_WORDS)
    newlist = [word.strip() for word in word_list if word not in stopword_dict]
    sentence = " ".join(newlist)
    
    sentence = url(sentence)
    sentence = num_repl(sentence)
    from string import punctuation
    #punctuations = '''@#!?+&*[]%.:/-();$=><|{}^'`\\'''
    punctuations = set(punctuation)
    for p in punctuations:
        sentence = sentence.replace(p, " ")
    sentence.strip()
    
    return sentence
train["text"] = train.TITLE + " " + train.ABSTRACT
test["text"] = test.TITLE + " " + test.ABSTRACT
text_repl = [("- -"," "),("--"," "),("-"," "),("_"," ")]
for old, new in text_repl:
    train.text = train.text.str.replace(old, new)
    test.text = test.text.str.replace(old, new)
train["clean_text"] = train.text.apply(remove_stop)
test["clean_text"] = test.text.apply(remove_stop)

train["clean_text"] = train.clean_text.apply(lambda x: " ".join([w for w in x.split() if (len(w)>2)]))
test["clean_text"] = test.clean_text.apply(lambda x: " ".join([w for w in x.split() if (len(w)>2)]))

train["clean_text"] = train.clean_text.apply(lambda x: " ".join([w for w in x.split() if (len(set(w))>2)]))
test["clean_text"] = test.clean_text.apply(lambda x: " ".join([w for w in x.split() if (len(set(w))>2)]))
train.head()
test.head()
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from sklearn.model_selection import train_test_split
from nltk.stem.snowball import EnglishStemmer
wnl = WordNetLemmatizer()
stemmer = EnglishStemmer()
def word_lem_stem(text, lemm = True):
    if lemm:
        normalized = " ".join([wnl.lemmatize(word) for word in text.split()])
        return normalized
    else:
        normalized = " ".join([stemmer.stem(word) for word in text.split()])
        return normalized
    
train["final_text"] = train.clean_text.apply(lambda x: word_lem_stem(x,True))
test["final_text"] = test.clean_text.apply(lambda x: word_lem_stem(x,True))
train.head()
test.head()
train["count_unique"] = train.clean_text.apply(lambda x: len(set(x.split())))
test["count_unique"] = test.clean_text.apply(lambda x: len(set(x.split())))

# add feature len of the text for each record
def save_sub(file):
    sample.to_csv("/kaggle/working/model_"+file+".csv", index=False)
def pred_model(model, x, y, train_size=0.7, seed=1):
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size=train_size, random_state = 1)
    #xtrain = vectorizer.fit_transform(xtrain)
    #cols = vectorizer.get_feature_names()
    #xtest = vectorizer.transform(xtest)
    model.fit(xtrain, ytrain)

    trainpred = model.predict(xtrain)
    testpred = model.predict(xtest)
    
    print("Train...")
    print(classification_report(ytrain, trainpred))
    print("Test...")
    print(classification_report(ytest, testpred))

def total_model(model, x, y, test, seed=1):
    #x = vectorizer.fit_transform(x)
    #cols = vectorizer.get_feature_names()
    #test = vectorizer.transform(test)
    model.fit(x, y)

    trainpred = model.predict(x)
    pred = model.predict(test)
    
    print("Train...")
    print(classification_report(y, trainpred))
    return(pred)
X = train["final_text"]
Y = train[target]
model_nb = MultinomialNB()
model_rf = RandomForestClassifier(random_state=1, n_jobs=8)
model_logr = LogisticRegression(random_state=1, n_jobs=8)
model_lgbm = LGBMClassifier(random_state=1, n_jobs=8)
model_svc = SVC()
cv = CountVectorizer()
tfidf = TfidfVectorizer()
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline, Pipeline
model_xgb = XGBClassifier(random_state=1, n_jobs=8)#, n_estimators=100, max_depth=6, reg_alpha=0.1)
model_multi = MultiOutputClassifier(model_xgb, n_jobs=8)

pipe = Pipeline([('countvector',cv),('multi',model_multi)])
pipe.named_steps
X = train["final_text"]
Y = train[target]
pred_model(pipe, X, Y)
pred = total_model(pipe, X, Y, test.final_text)
sample.iloc[:,1:] = pred
sample.iloc[:,1:].sum()
save_sub('xgb')

