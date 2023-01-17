import os

os.listdir()
import tensorflow as tf

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from matplotlib import rcParams

import seaborn as sns

import re

import nltk

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score

from sklearn.metrics import recall_score, precision_score

from sklearn.metrics import confusion_matrix

import seaborn as sns



pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)

np.set_printoptions(threshold=np.inf)

rcParams['figure.figsize'] = (15,15)
df = pd.read_json("../input/train.json", orient='records')
# printing head of dataframe

df.head(10)
print("number of records {0}".format(len(df.id)))
# printing cuisine types and counts

print(df.cuisine.value_counts(normalize=True))

print("Number of cuisine types {0}".format(len(df.cuisine.value_counts())))

print("number of unique ingredients {0}".format(len(set([ingredient for ingredient_list in df.ingredients.values for ingredient in ingredient_list]))))
# plotting countplot

sns.countplot(y='cuisine',data=df)

plt.show()
def text_prepare(ingredient):

    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')

    BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')

    DOUBLE_SPACE_RE = re.compile('\s{2,}')

    STOPWORDS = set(stopwords.words('english'))

    

    ingredient = ingredient.strip(" ")

    #lower casing letters

    ingredient = ingredient.lower()

    #replacing symbols by space

    ingredient = re.sub(REPLACE_BY_SPACE_RE,' ',ingredient)

    #deleting bad words

    ingredient = re.sub(BAD_SYMBOLS_RE,'',ingredient)

    #removing double space

    ingredient = re.sub(DOUBLE_SPACE_RE,'',ingredient)

    # remove numbers with percentages

    ingredient = re.sub('[0-9]*% ','',ingredient)

    # remove ounce information

    ingredient = re.sub("\(.*oz.\)",'',ingredient)

    # remove brand names with registered

    ingredient = re.sub("[A-Z]*[a-z]*®",'',ingredient)

    # remove brand names with trademark

    ingredient = re.sub("[A-Z]*[a-z]*™", '', ingredient)

    # remove numbers with +

    ingredient = re.sub("[0-9]+",'',ingredient)

    # replace & and -

    ingredient = ingredient.replace("&",'')

    ingredient = ingredient.replace("-", '')

    # lowercase all indegredients

    ingredient = ingredient.lower()

    # removing whitespacing once more

    ingredient = ingredient.strip()

    #removing stop words

    ingredient = ' '.join([word for word in ingredient.split(" ") if word not in STOPWORDS])

    

    return ingredient

def apply_text_prepare_to_list(ingredient_list):

    return ' '.join([text_prepare(ingredient) for ingredient in ingredient_list])
df.ingredients = df.ingredients.apply(apply_text_prepare_to_list)
print(df.head())
DICT_SIZE = 10000

def tfidf():

    tfidf_vectorizer = TfidfVectorizer(token_pattern=r'(\S+)',ngram_range=(1,2),strip_accents='unicode',max_features=DICT_SIZE)

    return tfidf_vectorizer

    

def vectorize(X_train,X_val,X_test,vectorizer):

    print("About to vectorize lmao",X_train.shape,X_val.shape,X_test.shape)

    X_train = vectorizer.fit_transform(X_train)

    X_val = vectorizer.transform(X_val)

    X_test = vectorizer.transform(X_test)

    return X_train.toarray(),X_val.toarray(),X_test.toarray(),vectorizer.vocabulary_



def bigram():

    bigram_vectorizer = CountVectorizer(ngram_range=(1,2),strip_accents='unicode',max_features=DICT_SIZE)

    return bigram_vectorizer
X_train, X_test, y_train, y_test = train_test_split(df.ingredients.values,df.cuisine.values,train_size=0.8,test_size=0.2,random_state=2019)

X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,train_size=0.8,test_size=0.2,random_state=42)

print(X_train.shape,X_val.shape,X_test.shape,y_train.shape,y_val.shape,y_test.shape)

print(X_train[0],"\n",y_train[0])
X_train_bow,X_val_bow,X_test_bow,bow_vocab = vectorize(X_train,X_val,X_test,bigram())

X_train_tfidf,X_val_tfidf,X_test_tfidf,tfidf_vocab = vectorize(X_train,X_val,X_test,tfidf())
print(X_train_bow.shape,X_val_bow.shape,X_test_bow.shape,X_train_tfidf.shape,X_val_tfidf.shape,X_test_tfidf.shape)

print(X_train_tfidf[0][0:10],"\n",X_train_bow[0][0:10])
def logistic_classifier():

    log_clf = LogisticRegression(random_state=2019,max_iter=100,solver='liblinear')

    return log_clf



def mlp_classifier():

    mlp_clf = MLPClassifier(hidden_layer_sizes=(8,16,8),batch_size=16,random_state=2019)

    return mlp_clf



def train(X,y,clf):

    print("fitting X y with shapes {0} and {1}".format(X.shape,y.shape))

    clf.fit(X,y)

    print(clf.score(X,y))

    return clf



def predict(X,clf):

    print("predicting X y with shapes {0}".format(X.shape))

    return clf.predict(X)



def evaluate(y_val,predicted):

    # accuracy

    # f1 score

    # confusion matrix

    # precision score

    # recall score

    print("accuracy:- {0}".format(accuracy_score(y_val,predicted)))

    prec_val = precision_score(y_val,predicted,average='macro')

    rec_val = recall_score(y_val,predicted,average='macro')

    f1_val = f1_score(y_val,predicted,average='macro')

    print("precision score :- {0} \n recall score :- {1} \n f1_score:- {2}".format(prec_val,rec_val,f1_val))

    conf_mat = confusion_matrix(y_val,predicted)

    sns.heatmap(conf_mat)

    plt.show()

    

    

def pipeline(X_train,y_train,X_val,y_val,X_test,y_test,clf):

    print("Begin Training")

    train(X_train,y_train,clf)

    print("Finished Training")

    print("Begin Prediction on validation set")

    y_val_pred = predict(X_val,clf)

    print("Begin evaluation on validation set")

    evaluate(y_val,y_val_pred)

    print("Begin prediction on test set")

    y_test_pred = predict(X_test,clf)

    print("Begin evaluation on test set")

    evaluate(y_test,y_test_pred)

    
pipeline(X_train_bow,y_train,X_val_bow,y_val,X_test_bow,y_test,logistic_classifier())
pipeline(X_train_bow,y_train,X_val_bow,y_val,X_test_bow,y_test,mlp_classifier())
pipeline(X_train_tfidf,y_train,X_val_tfidf,y_val,X_test_tfidf,y_test,logistic_classifier())
pipeline(X_train_tfidf,y_train,X_val_tfidf,y_val,X_test_tfidf,y_test,mlp_classifier())