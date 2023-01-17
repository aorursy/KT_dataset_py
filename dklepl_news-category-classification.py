import numpy as np

import pandas as pd

import os

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import KFold, cross_validate, train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV



import matplotlib.pyplot as plt

import seaborn as sns
PATH = '../input/news-category-dataset/News_Category_Dataset_v2.json'
data = pd.read_json(PATH, lines=True)

data.head(5)
data = data.drop(['authors', 'link', 'date'], axis=1)

data.head(5)
data['text'] = data['headline'] + " " + data['short_description']

data.head(5)
data = data.drop(['headline', 'short_description'], axis=1)

data.head(5)
categories = data.groupby('category')

print("There's {} news categories." .format(categories.ngroups))

print(categories.size())
culture = ['ARTS & CULTURE', 'ARTS', 'CULTURE & ARTS']

home_living = ['HOME & LIVING', 'HEALTHY LIVING', 'WELLNESS']

style = ['STYLE', 'STYLE & BEAUTY']

entertainment = ['COMEDY', 'ENTERTAINMENT', 'MEDIA']

business = ['MONEY', 'BUSINESS']

parenting = ['PARENTING', 'PARENTS']

science_tech = ['SCIENCE', 'TECH']

education = ['COLLEGE', 'EDUCATION']

drop = ['BLACK VOICES', 'DIVORCE', 'FIFTY', 'GOOD NEWS', 'IMPACT', 'LATINO VOICES', 'WOMEN', 

        'QUEER VOICES', 'TASTE', 'THE WORLDPOST', 'WORLD NEWS', 'WORLDPOST', 'GREEN', 'WEIRD NEWS', 'DIVORCE', ]
data_improved = data[~data.category.isin(drop)]
data_improved.category[data_improved.category.isin(culture)] = "CULTURE"

data_improved.category[data_improved.category.isin(home_living)] = "HOME & LIVING"

data_improved.category[data_improved.category.isin(style)] = "STYLE"

data_improved.category[data_improved.category.isin(entertainment)] = "ENTERTAINMENT"

data_improved.category[data_improved.category.isin(business)] = "BUSINESS"

data_improved.category[data_improved.category.isin(parenting)] = "PARENTING"

data_improved.category[data_improved.category.isin(education)] = "EDUCATION"

data_improved.category[data_improved.category.isin(science_tech)] = "SCIENCE & TECH"
data_improved = data_improved.reset_index(drop=True)
len(data_improved)
data_improved.text = data_improved.text.replace('[0-9]', '', regex=True)
lenght = []

for i in range(len(data_improved)):

    lenght.append(len(data_improved.text[i].split()))
data_improved['lenght'] = lenght
data_improved.head(5)
data_improved = data_improved[data_improved.lenght>10]

len(data_improved)
categories = data_improved.groupby('category')

print("There's {} news categories." .format(categories.ngroups))

print(categories.size())
fig, ax = plt.subplots(1, 1, figsize=(35,7))

sns.countplot(x = 'category', data = data_improved)
import spacy

from html import unescape



# create a spaCy tokenizer

spacy.load('en')

lemmatizer = spacy.lang.en.English()



#to lower, remove HTML tags

def my_preprocessor(doc):

    return(unescape(doc).lower())



# tokenize the doc and lemmatize its tokens

def my_tokenizer(doc):

    tokens = lemmatizer(doc)

    return([token.lemma_ for token in tokens])
X_train, X_test, y_train, y_test = train_test_split(data_improved.text, data_improved.category, test_size=0.2, random_state=42)
folds = KFold(n_splits = 5, shuffle = True, random_state = 1)
vect = TfidfVectorizer(stop_words='english', ngram_range = (1,2), max_features = 15000, preprocessor=my_preprocessor, tokenizer=my_tokenizer)
from sklearn.svm import LinearSVC

from sklearn.linear_model import RidgeClassifier

from xgboost.sklearn import XGBClassifier
NB = MultinomialNB()

Ridge = RidgeClassifier()

XGB = XGBClassifier(objective = 'multi:softprob')
def fit_model(clf , name):

    pipe = Pipeline([

    ('vectorize', vect),

    (name, clf)])

    result = cross_validate(pipe, X_train, y_train, cv = folds, return_train_score=True,scoring = ('accuracy', 

                                                                                       'f1_weighted', 

                                                                                       'precision_weighted', 

                                                                                       'recall_weighted'))

    return result
bayes = fit_model(NB, 'NB')

ridge = fit_model(Ridge, 'Ridge')

xgb = fit_model(XGB, 'XGB')
bayes.keys()
b = pd.DataFrame.from_dict(bayes)

b['model'] = ["NB","NB","NB","NB","NB"]

r = pd.DataFrame.from_dict(ridge)

r['model'] = ["Ridge","Ridge","Ridge","Ridge","Ridge"]

x = pd.DataFrame.from_dict(xgb)

x['model'] = ["XGB","XGB","XGB","XGB","XGB"]



results = pd.concat([b,r,x])
results.sample(6)
models_eval = results.groupby("model")
means = models_eval.mean()

sd = models_eval.std()
means.head()
sd.head()
from sklearn import metrics

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.base import clone

from sklearn.preprocessing import label_binarize

from scipy import interp

from sklearn.metrics import roc_curve, auc 





def get_metrics(true_labels, predicted_labels):

    

    print('Accuracy:', np.round(

                        metrics.accuracy_score(true_labels, 

                                               predicted_labels),

                        4))

    print('Precision:', np.round(

                        metrics.precision_score(true_labels, 

                                               predicted_labels,

                                               average='weighted'),

                        4))

    print('Recall:', np.round(

                        metrics.recall_score(true_labels, 

                                               predicted_labels,

                                               average='weighted'),

                        4))

    print('F1 Score:', np.round(

                        metrics.f1_score(true_labels, 

                                               predicted_labels,

                                               average='weighted'),

                        4))

                        



def train_predict_model(classifier, 

                        train_features, train_labels, 

                        test_features, test_labels):

    # build model    

    classifier.fit(train_features, train_labels)

    # predict using model

    predictions = classifier.predict(test_features) 

    return predictions    





def display_confusion_matrix(true_labels, predicted_labels, classes=[1,0]):

    

    total_classes = len(classes)

    level_labels = [total_classes*[0], list(range(total_classes))]



    cm = metrics.confusion_matrix(y_true=true_labels, y_pred=predicted_labels, 

                                  labels=classes)

    cm_frame = pd.DataFrame(data=cm, 

                            columns=pd.MultiIndex(levels=[['Predicted:'], classes], 

                                                  labels=level_labels), 

                            index=pd.MultiIndex(levels=[['Actual:'], classes], 

                                                labels=level_labels)) 

    print(cm_frame) 

    

def display_classification_report(true_labels, predicted_labels, classes=[1,0]):



    report = metrics.classification_report(y_true=true_labels, 

                                           y_pred=predicted_labels, 

                                           labels=classes) 

    print(report)

    

    

    

def model_perf(classes=y_test.unique()):

    print('Model Performance metrics:')

    print('-'*30)

    get_metrics(true_labels=true_labels, predicted_labels=predicted_labels)

    print('\nModel Classification report:')

    print('-'*30)

    display_classification_report(true_labels=true_labels, predicted_labels=predicted_labels, 

                                  classes=classes)

    print('\nPrediction Confusion Matrix:')

    print('-'*30)

    display_confusion_matrix(true_labels=true_labels, predicted_labels=predicted_labels, 

                             classes=classes)