import pandas as pd

from pandas import Series,DataFrame

import numpy as np
#data

df=pd.read_csv('../input/AllTweets.csv')
pd.DataFrame(df.groupby('author').size().rename('counts')).sort_values('counts', ascending=False)
import random

from sklearn.model_selection import train_test_split

#1000 random sample rows for each author

df_new=pd.DataFrame()

twts_train=pd.DataFrame()

twts_test=pd.DataFrame()

author_train=pd.DataFrame()

author_test=pd.DataFrame()

for a in df.author.unique():

    rows = random.sample(list(df[df['author']==a].index), 1000)

    df_temp = df.ix[rows]

    df_new=df_new.append(df_temp,ignore_index=True)    

    X_train, X_test, Y_train, Y_test = train_test_split(df_temp.ix[:,['text']], df_temp.ix[:,['author']], test_size=0.2, random_state=42)

    twts_train=twts_train.append(X_train, verify_integrity=False)

    twts_test=twts_test.append(X_test, verify_integrity=False)

    author_train=author_train.append(Y_train, verify_integrity=False)

    author_test=author_test.append(Y_test, verify_integrity=False)
print (len(twts_train),len(author_train))
print(len(twts_test),len(author_test))
from nltk.tokenize import RegexpTokenizer

from nltk.stem.porter import PorterStemmer

def text_process(text):

    """

    Takes in a string of text, then performs the following:

    1. Tokenizes and removes punctuation

    3. Stems

    4. Returns a list of the cleaned text

    """



    # tokenizing

    tokenizer = RegexpTokenizer(r'\w+')

    text_processed=tokenizer.tokenize(text)

    

    

    # steming

    porter_stemmer = PorterStemmer()

    

    text_processed = [porter_stemmer.stem(word) for word in text_processed]

    



    return text_processed
from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
ScoreSummaryByModel = list()
def PredictionEvaluation(author_test_b,author_predicted_b,target_names,comment):

    Accuracy=accuracy_score(author_test_b,author_predicted_b)

    #print (Accuracy)

    Recall=recall_score(author_test_b, author_predicted_b, labels=[0,1,2,3], average='macro')

    #print (Recall)

    Precision=precision_score(author_test_b, author_predicted_b, labels=[0,1,2,3], average='macro')

    #print (Precision)

    F1=f1_score(author_test_b, author_predicted_b, labels=[0,1,2,3], average='macro')

    #print (F1)

    ScoreSummaryByModel.append([Accuracy,Recall,Precision,F1,comment])

    print(classification_report(author_test_b, author_predicted_b, target_names=target_names))
import matplotlib.pyplot as plt

%matplotlib inline

import itertools
#http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py

def plot_confusion_matrix(cm, classes,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
ScoreSummaryByModelParams=list()
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.pipeline import FeatureUnion

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import LabelBinarizer
def ModelParamsEvaluation (f_union,model,params,comment):

    pipeline = Pipeline([

    # Extract the text & text_coded

    # Use FeatureUnion to combine the features from different vectorizers

    ('union', f_union),

    # Use a  classifier on the combined features

    ('clf', model)

    ])

    grid_search = GridSearchCV(estimator=pipeline, param_grid=params, verbose=1)

    grid_search.fit(twts_train['text'], author_train['author'])

    author_predicted = grid_search.predict(twts_test['text'])

    lb = LabelBinarizer()

    author_test_b = lb.fit_transform(author_test['author'])

    author_predicted_b  = lb.fit_transform(author_predicted)

    #best score

    print("Best score: %0.3f" % grid_search.best_score_)

    print("Best parameters set:")

    best_parameters = grid_search.best_estimator_.get_params()

    author_names=grid_search.best_estimator_.named_steps['clf'].classes_



    for param_name in sorted(params.keys()):

        print("\t%s: %r" % (param_name, best_parameters[param_name]))

        ScoreSummaryByModelParams.append([comment,grid_search.best_score_,"\t%s: %r" % (param_name, best_parameters[param_name])]) 

    return (author_predicted,author_predicted_b,author_test_b,author_names)
f2_union=FeatureUnion(

        transformer_list=[

            # Pipeline for pulling char features  from the text

            ('char', Pipeline([

                ('tfidf',     TfidfVectorizer(analyzer='char',ngram_range=(3, 3))),

            ])),

            # Pipeline for pulling stememd word features from the text

            ('text', Pipeline([

                ('tfidf',    TfidfVectorizer(analyzer='word',tokenizer= text_process,ngram_range=(1, 1))),

            ])),        



        ],



    )
from sklearn.svm import LinearSVC

#LinearSVC

p = {'clf__C': (1,0.1,0.01,0.001,0.0001)}

(author_predicted,author_predicted_b, author_test_b,author_names)=ModelParamsEvaluation(f2_union,LinearSVC(),p,'LinearSVC')
PredictionEvaluation(author_predicted_b, author_test_b,author_names,'LinearSVC')
plot_confusion_matrix(confusion_matrix(author_test['author'], author_predicted), author_names,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues)
author_predicted=pd.DataFrame(author_predicted,columns=['predicted'])

df_wrong_result = pd.concat([twts_test.reset_index(),author_test.reset_index(),author_predicted], axis=1)

df_wrong_result.drop('index', axis=1, inplace=True)

df_wrong_result=df_wrong_result[df_wrong_result['author']!=df_wrong_result['predicted']]

df_wrong_result.head(10)
df_wrong_result[df_wrong_result['predicted']=='KimKardashian'].head(20)
#from sklearn.svm import SVC

#p = {'clf__C': (1,0.1,0.01,0.001,0.0001)}

#(author_predicted,author_predicted_b, author_test_b,author_names)=ModelParamsEvaluation(f2_union,SVC(kernel='linear'),p,'SVC, linear kernel')
##PredictionEvaluation(author_predicted_b, author_test_b,author_names,'SVC, linear kernel')
#plot_confusion_matrix(confusion_matrix(author_test['author'], author_predicted), author_names,

#                          title='Confusion matrix',

#                         cmap=plt.cm.Blues)
from sklearn.linear_model import SGDClassifier

p = {'clf__alpha': (0.01,0.001,0.0001,0.00001, 0.000001),

    'clf__penalty': ('l1','l2', 'elasticnet')}

(author_predicted,author_predicted_b, author_test_b,author_names)=ModelParamsEvaluation (f2_union,SGDClassifier(),p,'SGD Classifier')
PredictionEvaluation(author_predicted_b, author_test_b,author_names,'SGD Classifier')
plot_confusion_matrix(confusion_matrix(author_test['author'], author_predicted), author_names,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues)
from sklearn.naive_bayes import BernoulliNB

p = {'clf__alpha': (1,0.1,0.01,0.001,0.0001,0)}

(author_predicted,author_predicted_b, author_test_b,author_names)=ModelParamsEvaluation(f2_union,BernoulliNB(),p,'Bernoulli Naive Bayes')
PredictionEvaluation(author_predicted_b, author_test_b,author_names,'Bernoulli Naive Bayes')
plot_confusion_matrix(confusion_matrix(author_test['author'], author_predicted), author_names,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues)
df_ScoreSummaryByModelParams=DataFrame(ScoreSummaryByModelParams,columns=['Method','BestScore','BestParameter'])

df_ScoreSummaryByModelParams.sort_values(['BestScore'],ascending=False,inplace=True)

df_ScoreSummaryByModelParams
df_ScoreSummaryByModel=DataFrame(ScoreSummaryByModel,columns=['Precision','Accuracy','Recall','F1','Comment'])

df_ScoreSummaryByModel.sort_values(['F1'],ascending=False,inplace=True)

df_ScoreSummaryByModel