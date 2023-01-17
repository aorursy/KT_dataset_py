print("Hello World")
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



import re

import sys

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_recall_fscore_support

from nltk.stem import LancasterStemmer

from nltk.tokenize import sent_tokenize, word_tokenize

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB, MultinomialNB



import warnings

warnings.filterwarnings("ignore")
train = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/train.csv")

#test= pd.read_csv("data//test.csv")
train.shape
train, test = train_test_split(train, test_size = 0.2)

train.shape, test.shape
train.to_csv("train_jigsaw.csv", index=None)

test.to_csv("test_jigsaw.csv", index=None)
train.shape
train.isna().sum()
test.isna().sum()
for data in [train, test]:

    print(data.apply(lambda x:len(x.unique())))
train.head()
train.columns
labels = ['toxic','severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

values = train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].sum()
plt.bar(labels, values);

plt.ylabel("Frequency");

plt.title("Comments Types");
full_text = [i for i in train['comment_text']] + [i for i in test['comment_text']] 
len(full_text), len(train) + len(test)
#ref https://medium.com/@datamonsters/text-preprocessing-in-python-steps-tools-and-examples-bf025f872908



def preprocess_text(text):

    text = text.lower()

    text = re.sub(r'\d+', '', text)

    text = re.sub('[^A-Za-z0-9]+', ' ', text)

    text = text.strip()

    return text
text = ['Hi How are you??', 'I am good !!! :D']
clean_text = preprocess_text(text[1])
clean_text = [(lambda x: preprocess_text(x))(x) for x in full_text]
len(clean_text)
train.shape
def bag_of_words(clean_text, x_train_text, x_test_text):

    count_vect = CountVectorizer(stop_words=set(stopwords.words('english')))

    count_vect.fit(clean_text)

    x_train_count_vec = count_vect.transform(x_train_text)

    x_test_count_vec = count_vect.transform(x_test_text)

    print(x_train_count_vec.shape, x_test_count_vec.shape)

    

    return x_train_count_vec, x_test_count_vec
x_train_count_vec, x_test_count_vec = bag_of_words(clean_text,clean_text[:127656], clean_text[127656:])
def tfidf_transform(clean_text, x_train_text, x_test_text):

    vectorizer = TfidfVectorizer(stop_words = set(stopwords.words('english')))

    vectorizer.fit(clean_text)

    x_train_tfidf_vec = vectorizer.transform(x_train_text)

    x_test_tfidf_vec = vectorizer.transform(x_test_text)

    print(x_train_tfidf_vec.shape, x_test_tfidf_vec.shape)

    return x_train_tfidf_vec, x_test_tfidf_vec  
x_train_tfidf_vec, x_test_tfidf_vec = tfidf_transform(clean_text, clean_text[:127656], clean_text[127656:])
clean_text[2]
def modelling(clf, x_train, y_train, x_test, y_test):

    

    #X_train, X_cv, y_train, y_cv = train_test_split( x_train, y_train, test_size=0.1, stratify = y_train,random_state=0)

    

    clf.fit(x_train, y_train)

    y_train_pred = clf.predict(x_train)

    

    y_test_pred = clf.predict(x_test)

    

    vals = precision_recall_fscore_support(y_test, y_test_pred, average='macro')

    precision = vals[0]

    recall = vals[1]

    f1 = vals[2]

    acc = accuracy_score(y_test, y_test_pred)

    print("accuracy: ", acc, f1)

    print("confusion matrix for CV is ")

    print(confusion_matrix(y_test, y_test_pred ))

    

    return y_train_pred, y_test_pred, precision, recall, f1, acc
def tune_parameters(clf, params, x_train, y_train):

    clf = GridSearchCV(clf, params, cv=5)

    clf.fit(x_train,y_train)

    print("best parameters for model are ",clf.best_params_)

    print("Accuracy is ",clf.best_score_)
params = {'C': [ 0.01, 0.1, 1, 10, 100] }

clf = LogisticRegression()

#tune_parameters(clf, params, x_train_tfidf_vec, train['severe_toxic'])
#tune_parameters(clf, params, x_train_tfidf_vec, train['toxic'])
#tune_parameters(clf, params, x_train_tfidf_vec, train['obscene'])
#print(tune_parameters(clf, params, x_train_tfidf_vec, train['threat']))

#print(tune_parameters(clf, params, x_train_tfidf_vec, train['insult']))

#print(tune_parameters(clf, params, x_train_tfidf_vec, train['identity_hate']))
c_dict = {'toxic': 10,

          'severe_toxic':1,

          'obscene':10,

          'threat':100,

          'insult':10,

          'identity_hate':10}
y_train_pred, y_test_pred, precision, recall, f1, acc = modelling(LogisticRegression(C = 1), 

                                                                            x_train_tfidf_vec, 

                                                                            train['severe_toxic'], 

                                                                            x_test_tfidf_vec,

                                                                             test['severe_toxic'])
pred_logistic_df = pd.DataFrame()



logistic_results_cv = pd.DataFrame({'labels': labels})

logistic_results_cv['acc'] = 0

logistic_results_cv['f1'] = 0

logistic_results_cv['precision'] = 0

logistic_results_cv['recall']  = 0





pred_logistic_df['id'] = test['id']



for col in labels:

    print("Modelling for: ",col)

    _, y_test_logistic, precision, recall, f1, acc  = modelling(LogisticRegression(C = c_dict[col]),

                                      x_train_count_vec,

                                      train[col], 

                                      x_test_count_vec, test[col])



    logistic_results_cv['acc'][logistic_results_cv['labels']==col] = acc

    logistic_results_cv['f1'][logistic_results_cv['labels']==col] = f1

    logistic_results_cv['precision'][logistic_results_cv['labels']==col] = precision

    logistic_results_cv['recall'][logistic_results_cv['labels']==col] = recall

    

    pred_logistic_df[col] =  y_test_logistic

    

    print("\n")
logistic_results_cv
pred_logistic_df.head()
logistic_results_cv.to_csv("logistic_bag_words.csv", index=None, header=True)
pred_logistic_bag_df = pd.DataFrame()



logistic_results_cv = pd.DataFrame({'labels': labels})

logistic_results_cv['acc'] = 0

logistic_results_cv['f1'] = 0

logistic_results_cv['precision'] = 0

logistic_results_cv['recall']  = 0





pred_logistic_bag_df['id'] = test['id']



for col in labels:

    print("Modelling for: ",col)

    _, y_test_logistic, precision, recall, f1, acc  = modelling(LogisticRegression(C = c_dict[col]),

                                      x_train_tfidf_vec,

                                      train[col], 

                                      x_test_tfidf_vec, test[col])



    logistic_results_cv['acc'][logistic_results_cv['labels']==col] = acc

    logistic_results_cv['f1'][logistic_results_cv['labels']==col] = f1

    logistic_results_cv['precision'][logistic_results_cv['labels']==col] = precision

    logistic_results_cv['recall'][logistic_results_cv['labels']==col] = recall

    

    pred_logistic_bag_df[col] =  y_test_logistic

    

    print("\n")
logistic_results_cv
logistic_results_cv.to_csv("logistic_bag_first.csv", index=None, header=True)
rf = RandomForestClassifier(random_state=42)



params = { 

    'n_estimators': [50, 200],

    'max_features': ['auto', 'sqrt', 'log2'],

    'max_depth' : [5,10,15,30],

    'criterion' :['gini', 'entropy']

}

#tune_parameters(rf, params, x_train_tfidf_vec, train['toxic'])
pred_rf_bag_df = pd.DataFrame()



rf_results_cv = pd.DataFrame({'labels': labels})

rf_results_cv['acc'] = 0

rf_results_cv['f1'] = 0

rf_results_cv['precision'] = 0

rf_results_cv['recall']  = 0



pred_rf_bag_df['id'] = test['id']

for col in labels:

    print("Modelling for: ",col)

    _, y_test_rf, precision, recall, f1, acc  = modelling(RandomForestClassifier(),

                                      x_train_count_vec,

                                      train[col], 

                                      x_test_count_vec, test[col])



    rf_results_cv['acc'][rf_results_cv['labels']==col] = acc

    rf_results_cv['f1'][rf_results_cv['labels']==col] = f1

    rf_results_cv['precision'][rf_results_cv['labels']==col] = precision

    rf_results_cv['recall'][rf_results_cv['labels']==col] = recall

    

    pred_rf_bag_df[col] =  y_test_rf

    print("\n")
rf_results_cv
rf_results_cv.to_csv("rf_bag_first.csv", index=None, header=True)

pred_rf_bag_df.to_csv("pred_rf_bag_df_first.csv", index=None, header=True)
pred_rf_tfidf_df = pd.DataFrame()



rf_results_cv = pd.DataFrame({'labels': labels})

rf_results_cv['acc'] = 0

rf_results_cv['f1'] = 0

rf_results_cv['precision'] = 0

rf_results_cv['recall']  = 0





pred_rf_tfidf_df['id'] = test['id']



for col in labels:

    print("Modelling for: ",col)

    _, y_test_rf, precision, recall, f1, acc  = modelling(RandomForestClassifier(),

                                      x_train_tfidf_vec,

                                      train[col], 

                                      x_test_tfidf_vec, test[col])



    rf_results_cv['acc'][rf_results_cv['labels']==col] = acc

    rf_results_cv['f1'][rf_results_cv['labels']==col] = f1

    rf_results_cv['precision'][rf_results_cv['labels']==col] = precision

    rf_results_cv['recall'][rf_results_cv['labels']==col] = recall

    

    pred_rf_tfidf_df[col] =  y_test_rf

    

    print("\n")
rf_results_cv
rf_results_cv.to_csv("rf_tfidf_first.csv", index=None, header=True)

pred_rf_tfidf_df.to_csv("pred_rf_tfidf_df_first.csv", index=None, header=True)
pred_NB_bag_df = pd.DataFrame()



rf_results_cv = pd.DataFrame({'labels': labels})

rf_results_cv['acc'] = 0

rf_results_cv['f1'] = 0

rf_results_cv['precision'] = 0

rf_results_cv['recall']  = 0





pred_NB_bag_df['id'] = test['id']



for col in labels:

    print("Modelling for: ",col)

    _, y_test_NB, precision, recall, f1, acc  = modelling(MultinomialNB(),

                                      x_train_count_vec,

                                      train[col], 

                                      x_test_count_vec, test[col])



    rf_results_cv['acc'][rf_results_cv['labels']==col] = acc

    rf_results_cv['f1'][rf_results_cv['labels']==col] = f1

    rf_results_cv['precision'][rf_results_cv['labels']==col] = precision

    rf_results_cv['recall'][rf_results_cv['labels']==col] = recall

    

    pred_NB_bag_df[col] =  y_test_NB

    

    print("\n")
rf_results_cv
rf_results_cv.to_csv("NB_bag_first.csv", index=None, header=True)

pred_NB_bag_df.to_csv("pred_NB_bag_df.csv", index=None, header=True)
pred_NB_tfidf_df = pd.DataFrame()



rf_results_cv = pd.DataFrame({'labels': labels})

rf_results_cv['acc'] = 0

rf_results_cv['f1'] = 0

rf_results_cv['precision'] = 0

rf_results_cv['recall']  = 0





pred_NB_tfidf_df['id'] = test['id']



for col in labels:

    print("Modelling for: ",col)

    _, y_test_NB, precision, recall, f1, acc  = modelling(MultinomialNB(),

                                      x_train_tfidf_vec,

                                      train[col], 

                                      x_test_tfidf_vec, test[col])



    rf_results_cv['acc'][rf_results_cv['labels']==col] = acc

    rf_results_cv['f1'][rf_results_cv['labels']==col] = f1

    rf_results_cv['precision'][rf_results_cv['labels']==col] = precision

    rf_results_cv['recall'][rf_results_cv['labels']==col] = recall

    

    pred_NB_tfidf_df[col] =  y_test_rf

    print("\n")
rf_results_cv
rf_results_cv.to_csv("NB_tfidf_first.csv", index=None, header=True)

pred_NB_tfidf_df.to_csv("pred_NB_tfidf_df.csv", index=None, header=True)
#ref https://stackoverflow.com/questions/43018030/replace-apostrophe-short-words-in-python



contractions = {

"ain't": "am not ",

"aren't": "are not",

"can't": "cannot",

"can't've": "cannot have",

"'cause": "because",

"could've": "could have",

"couldn't": "could not",

"couldn't've": "could not have",

"didn't": "did not",

"doesn't": "does not",

"don't": "do not",

"hadn't": "had not",

"hadn't've": "had not have",

"hasn't": "has not",

"haven't": "have not",

"he'd": "he would",

"he'd've": "he would have",

"he'll": "he will",

"he'll've": "he shall have",

"he's": "he is",

"how'd": "how did",

"how'd'y": "how do you",

"how'll": "how will",

"how's": "how is",

"i'd": "I would",

"i'd've": "I would have",

"i'll": "I will",

"i'll've": "I shall have",

"i'm": "I am",

"i've": "I have",

"isn't": "is not",

"it'd": "it would",

"it'd've": "it would have",

"it'll": "it will",

"it'll've": "it will have",

"it's": "it is",

"let's": "let us",

"ma'am": "madam",

"mayn't": "may not",

"might've": "might have",

"mightn't": "might not",

"mightn't've": "might not have",

"must've": "must have",

"mustn't": "must not",

"mustn't've": "must not have",

"needn't": "need not",

"needn't've": "need not have",

"o'clock": "of the clock",

"oughtn't": "ought not",

"oughtn't've": "ought not have",

"shan't": "shall not",

"sha'n't": "shall not",

"shan't've": "shall not have",

"she'd": "she would",

"she'd've": "she would have",

"she'll": "she will",

"she'll've": "she will have",

"she's": "she is",

"should've": "should have",

"shouldn't": "should not",

"shouldn't've": "should not have",

"so've": "so have",

"so's": "so is",

"that'd": "that would",

"that'd've": "that would have",

"that's": "that is",

"there'd": "there would",

"there'd've": "there would have",

"there's": "there is",

"they'd": "they would",

"they'd've": "they would have",

"they'll": "they will",

"they'll've": "they will have",

"they're": "they are",

"they've": "they have",

"to've": "to have",

"wasn't": "was not",

"we'd": "we would",

"we'd've": "we would have",

"we'll": "we will",

"we'll've": "we will have",

"we're": "we are",

"we've": "we have",

"weren't": "were not",

"what'll": "what will",

"what'll've": "what will have",

"what're": "what are",

"what's": "what is",

"what've": "what have",

"when's": "when is",

"when've": "when have",

"where'd": "where did",

"where's": "where is",

"where've": "where have",

"who'll": "who will",

"who'll've": "who will have",

"who's": "who is",

"who've": "who have",

"why's": "why is",

"why've": "why have",

"will've": "will have",

"won't": "will not",

"won't've": "will not have",

"would've": "would have",

"wouldn't": "would not",

"wouldn't've": "would not have",

"y'all": "you all",

"y'all'd": "you all would",

"y'all'd've": "you all would have",

"y'all're": "you all are",

"y'all've": "you all have",

"you'd": "you would",

"you'd've": "you would have",

"you'll": "you will",

"you'll've": "you will have",

"you're": "you are",

"you've": "you have"

}

def extra_preprocess_text(text):    

    for word in text.split():

        if word.lower() in contractions:

            text = text.replace(word, contractions[word.lower()])    

    text = re.sub(r'\d+', '', text)

    text = re.sub('[^A-Za-z0-9]+', ' ', text)

    text = text.strip()

    return text
lancaster=LancasterStemmer()



sys.setrecursionlimit(1500)



def extra_preprocess_text_with_stemmer(text):    

    for word in text.split():

        if word.lower() in contractions:

            text = text.replace(word, contractions[word.lower()])    

            

        text = text.replace(word, lancaster.stem(word))

    text = re.sub(r'\d+', '', text)

    text = re.sub('[^A-Za-z0-9]+', ' ', text)

    text = text.strip()

    return text
preprocess_text('That\'s the beauty I am playing cricket running swimming :::????')
extra_preprocess_text('That\'s the beauty I am playing cricket running swimming :::????')
extra_preprocess_text_with_stemmer('That\'s the beauty I am playing cricket running swimming :::????')
extra_clean_text = [(lambda x: extra_preprocess_text(x))(x) for x in full_text]
extra_clean_text_with_stemming = [(lambda x: extra_preprocess_text_with_stemmer(x))(x) for x in full_text]
extra_clean_text[2]
clean_text[2]
extra_clean_text_with_stemming[2]
x_train_count_stemmed, x_test_count_stemmed = bag_of_words(extra_clean_text_with_stemming,

                                                   extra_clean_text_with_stemming[:127656], 

                                                   extra_clean_text_with_stemming[127656:])



x_train_tfidf_stemmed, x_test_tfidf_stemmed = tfidf_transform(extra_clean_text_with_stemming, 

                                                      extra_clean_text_with_stemming[:127656], 

                                                      extra_clean_text_with_stemming[127656:])
pred_NB_tfidf_df = pd.DataFrame()



rf_results_cv = pd.DataFrame({'labels': labels})

rf_results_cv['acc'] = 0

rf_results_cv['f1'] = 0

rf_results_cv['precision'] = 0

rf_results_cv['recall']  = 0





pred_NB_tfidf_df['id'] = test['id']



for col in labels:

    print("Modelling for: ",col)

    _, y_test_NB, precision, recall, f1, acc  = modelling(LogisticRegression(C = c_dict[col]), 

                                      x_train_count_stemmed,

                                      train[col], 

                                      x_test_count_stemmed, test[col])



    rf_results_cv['acc'][rf_results_cv['labels']==col] = acc

    rf_results_cv['f1'][rf_results_cv['labels']==col] = f1

    rf_results_cv['precision'][rf_results_cv['labels']==col] = precision

    rf_results_cv['recall'][rf_results_cv['labels']==col] = recall

    

    pred_NB_tfidf_df[col] =  y_test_rf

    print("\n")
rf_results_cv
pred_NB_tfidf_df = pd.DataFrame()



rf_results_cv = pd.DataFrame({'labels': labels})

rf_results_cv['acc'] = 0

rf_results_cv['f1'] = 0

rf_results_cv['precision'] = 0

rf_results_cv['recall']  = 0





pred_NB_tfidf_df['id'] = test['id']



for col in labels:

    print("Modelling for: ",col)

    _, y_test_NB, precision, recall, f1, acc  = modelling(LogisticRegression(C = c_dict[col]),

                                      x_train_tfidf_stemmed,

                                      train[col], 

                                      x_test_tfidf_stemmed, test[col])



    rf_results_cv['acc'][rf_results_cv['labels']==col] = acc

    rf_results_cv['f1'][rf_results_cv['labels']==col] = f1

    rf_results_cv['precision'][rf_results_cv['labels']==col] = precision

    rf_results_cv['recall'][rf_results_cv['labels']==col] = recall

    

    pred_NB_tfidf_df[col] =  y_test_rf

    print("\n")
rf_results_cv