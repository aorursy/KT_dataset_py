import pandas as pd

from pandas import Series,DataFrame

import numpy as np
#data

df=pd.read_csv('../input/techcrunch_posts.csv')
pd.DataFrame(df.groupby('authors').size().rename('counts')).sort_values('counts', ascending=False).head(10)
authors=['Sarah Perez','Anthony Ha','Darrell Etherington','Jordan Crook']
df=df[df['authors'].isin(authors)].ix[:,['authors','content']]

df.rename(columns={'authors':'author'}, inplace=True)

len(df)
import random

from sklearn.model_selection import train_test_split

#200 random sample rows for each author

df_new=pd.DataFrame()

posts_train=pd.DataFrame()

posts_test=pd.DataFrame()

author_train=pd.DataFrame()

author_test=pd.DataFrame()

for a in df.author.unique():

    rows = random.sample(list(df[df['author']==a].index), 200)

    df_temp = df.ix[rows]

    df_new=df_new.append(df_temp,ignore_index=True)    

    X_train, X_test, Y_train, Y_test = train_test_split(df_temp.ix[:,['content']], df_temp.ix[:,['author']], test_size=0.3, random_state=42)

    posts_train=posts_train.append(X_train, verify_integrity=False)

    posts_test=posts_test.append(X_test, verify_integrity=False)

    author_train=author_train.append(Y_train, verify_integrity=False)

    author_test=author_test.append(Y_test, verify_integrity=False)
print (len(posts_train),len(author_train))
print(len(posts_test),len(author_test))
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
ScoreSummaryByModelParams=list()
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.pipeline import FeatureUnion

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV
def ModelParamsEvaluation (f_union,model,params,comment):

    pipeline = Pipeline([

    # Extract the text & text_coded

    # Use FeatureUnion to combine the features from different vectorizers

    ('union', f_union),

    # Use a  classifier on the combined features

    ('clf', model)

    ])

    grid_search = GridSearchCV(estimator=pipeline, param_grid=params, verbose=1)

    grid_search.fit(posts_train['content'], author_train['author'])

    #best score

    print("Best score: %0.3f" % grid_search.best_score_)

    print("Best parameters set:")

    best_parameters = grid_search.best_estimator_.get_params()

    for param_name in sorted(params.keys()):

        print("\t%s: %r" % (param_name, best_parameters[param_name]))

        ScoreSummaryByModelParams.append([comment,grid_search.best_score_,"\t%s: %r" % (param_name, best_parameters[param_name])])    
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

ModelParamsEvaluation(f2_union,LinearSVC(),p,'LinearSVC')
from sklearn.svm import SVC

p = {'clf__C': (1,0.1,0.01,0.001,0.0001)}

ModelParamsEvaluation(f2_union,SVC(kernel='linear'),p,'SVC, linear kernel')
from sklearn.linear_model import SGDClassifier

p = {'clf__alpha': (0.01,0.001,0.0001,0.00001, 0.000001),

    'clf__penalty': ('l1','l2', 'elasticnet')}

ModelParamsEvaluation (f2_union,SGDClassifier(),p,'SGD Classifier')
from sklearn.naive_bayes import BernoulliNB

p = {'clf__alpha': (1,0.1,0.01,0.001,0.0001,0)}

ModelParamsEvaluation(f2_union,BernoulliNB(),p,'Bernoulli Naive Bayes')
df_ScoreSummaryByModelParams=DataFrame(ScoreSummaryByModelParams,columns=['Method','BestScore','BestParameter'])

df_ScoreSummaryByModelParams.sort_values(['BestScore'],ascending=False,inplace=True)

df_ScoreSummaryByModelParams
from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
ScoreSummaryByVector = list()
def PredictionEvaluation(author_test_b,author_predicted_b,target_names,comment):

    Accuracy=accuracy_score(author_test_b,author_predicted_b)

    Recall=recall_score(author_test_b, author_predicted_b, labels=[0,1,2,3], average='macro')

    Precision=precision_score(author_test_b, author_predicted_b, labels=[0,1,2,3], average='macro')

    F1=f1_score(author_test_b, author_predicted_b, labels=[0,1,2,3], average='macro')

    ScoreSummaryByVector.append([Accuracy,Recall,Precision,F1,comment])

    print(classification_report(author_test_b, author_predicted_b, target_names=target_names))
import matplotlib.pyplot as plt

%matplotlib inline

import itertools
#http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py

def plot_confusion_matrix(cm, classes,

                          normalize=False,

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



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
def ModelRun (f_union,model):

    pipeline = Pipeline([

    # Extract the text & text_coded

    # Use FeatureUnion to combine the features from different vectorizers

    ('union', f_union),

    # Use a  classifier on the combined features

    ('clf', model)

    ])

    

    pipeline.fit(posts_train['content'], author_train['author'])

    

    author_predicted = pipeline.predict(posts_test['content'])

    

    feature_names=list()

    for p in (pipeline.get_params()['union'].transformer_list):

        fn=(p[0],pipeline.get_params()['union'].get_params()[p[0]].get_params()['tfidf'].get_feature_names())

        feature_names.append(fn)

    df_fn=pd.DataFrame()

    for fn in feature_names:

        df_fn= df_fn.append(pd.DataFrame(

        {'FeatureType': fn[0],

         'Feature': fn[1]

        }),

        ignore_index=True)    

    

    from sklearn.preprocessing import LabelBinarizer

    lb = LabelBinarizer()

    author_test_b = lb.fit_transform(author_test['author'])

    author_predicted_b  = lb.fit_transform(author_predicted)

    return (df_fn,pipeline.get_params()['clf'],author_predicted,author_predicted_b, author_test_b)
(feature_names,clf,author_predicted,author_predicted_b, author_test_b)=ModelRun(f2_union,BernoulliNB(alpha=0.0001))

target_names=clf.classes_
PredictionEvaluation(author_predicted_b, author_test_b,target_names,'BernoulliNB(alpha=0.0001)')
plot_confusion_matrix(confusion_matrix(author_test['author'], author_predicted), target_names,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues)
(feature_names,clf,author_predicted,author_predicted_b, author_test_b)=ModelRun(f2_union,LinearSVC(C=1))

target_names=clf.classes_
PredictionEvaluation(author_predicted_b, author_test_b,target_names,'LinearSVC(C=1)')
plot_confusion_matrix(confusion_matrix(author_test['author'], author_predicted), target_names,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues)
df_ScoreSummaryByVector=DataFrame(ScoreSummaryByVector,columns=['Precision','Accuracy','Recall','F1','Comment'])

df_ScoreSummaryByVector.sort_values(['F1'],ascending=False,inplace=True)

df_ScoreSummaryByVector